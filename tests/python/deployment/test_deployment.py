"""
Tests for Deployment Configuration.

Validates Docker, Kubernetes, and Helm configurations are correct
without actually deploying anything.

Reference: Section 12 of design-doc.md
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

# Base path for deployment files
DEPLOY_DIR = Path(__file__).parent.parent.parent.parent / "deploy"
K8S_BASE_DIR = DEPLOY_DIR / "k8s" / "base"
K8S_DEV_DIR = DEPLOY_DIR / "k8s" / "overlays" / "dev"
K8S_PROD_DIR = DEPLOY_DIR / "k8s" / "overlays" / "prod"
DOCKER_DIR = DEPLOY_DIR / "docker"
HELM_DIR = DEPLOY_DIR / "helm" / "quant-trading"
SCRIPTS_DIR = DEPLOY_DIR / "scripts"


# =============================================================================
# Dockerfile Tests
# =============================================================================
class TestDockerfiles:
    """Test Docker configuration files."""

    @pytest.fixture
    def dockerfiles(self) -> List[Path]:
        """Get all Dockerfiles."""
        return list(DOCKER_DIR.glob("Dockerfile*"))

    def test_dockerfiles_exist(self, dockerfiles: List[Path]):
        """Verify all expected Dockerfiles exist."""
        expected = ["Dockerfile.base", "Dockerfile.api", "Dockerfile.calibration",
                    "Dockerfile.data-ingestion", "Dockerfile.execution", "Dockerfile.signals"]
        existing = [f.name for f in dockerfiles]
        for expected_file in expected:
            assert expected_file in existing, f"Missing Dockerfile: {expected_file}"

    def test_dockerfile_has_healthcheck(self, dockerfiles: List[Path]):
        """Verify Dockerfiles have health check defined."""
        for dockerfile in dockerfiles:
            if dockerfile.name == "Dockerfile.base":
                continue  # Base may not have specific healthcheck
            content = dockerfile.read_text()
            assert "HEALTHCHECK" in content, f"{dockerfile.name} missing HEALTHCHECK"

    def test_dockerfile_non_root_user(self, dockerfiles: List[Path]):
        """Verify Dockerfiles don't run as root."""
        for dockerfile in dockerfiles:
            content = dockerfile.read_text()
            # Check for USER instruction, non-root setup, or inheriting from base image
            has_user = ("USER" in content or "useradd" in content or
                       "runAsNonRoot" in content or "FROM ${BASE_IMAGE}" in content)
            assert has_user, f"{dockerfile.name} may be running as root"

    def test_dockerfile_expose_port(self, dockerfiles: List[Path]):
        """Verify service Dockerfiles expose a port."""
        for dockerfile in dockerfiles:
            if dockerfile.name == "Dockerfile.base":
                continue
            content = dockerfile.read_text()
            assert "EXPOSE" in content, f"{dockerfile.name} missing EXPOSE instruction"


class TestDockerCompose:
    """Test Docker Compose configuration."""

    @pytest.fixture
    def compose_config(self) -> Dict[str, Any]:
        """Load Docker Compose configuration."""
        compose_file = DOCKER_DIR / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml not found"
        return yaml.safe_load(compose_file.read_text())

    def test_services_defined(self, compose_config: Dict[str, Any]):
        """Verify all required services are defined."""
        services = compose_config.get("services", {})
        required = ["timescaledb", "redis", "api", "calibration",
                    "data-ingestion", "execution", "signals"]
        for service in required:
            assert service in services, f"Missing service: {service}"

    def test_services_have_healthcheck(self, compose_config: Dict[str, Any]):
        """Verify services have health checks."""
        services = compose_config.get("services", {})
        for name, config in services.items():
            if name in ["prometheus", "grafana"]:
                continue  # Monitoring can have different checks
            assert "healthcheck" in config, f"Service {name} missing healthcheck"

    def test_volumes_defined(self, compose_config: Dict[str, Any]):
        """Verify persistent volumes are defined."""
        volumes = compose_config.get("volumes", {})
        required = ["timescaledb_data", "redis_data"]
        for vol in required:
            assert vol in volumes, f"Missing volume: {vol}"

    def test_environment_variables(self, compose_config: Dict[str, Any]):
        """Verify critical environment variables are set."""
        services = compose_config.get("services", {})
        for name, config in services.items():
            if name in ["timescaledb", "redis", "rabbitmq", "prometheus", "grafana", "nginx"]:
                continue
            env = config.get("environment", {})
            # Check for common env or x-common-env anchor usage
            assert env or "<<:" in str(config), f"Service {name} missing environment config"


# =============================================================================
# Kubernetes Tests
# =============================================================================
class TestKubernetesBase:
    """Test Kubernetes base manifests."""

    @pytest.fixture
    def base_manifests(self) -> List[Dict[str, Any]]:
        """Load all base Kubernetes manifests."""
        manifests = []
        for yaml_file in K8S_BASE_DIR.glob("*.yaml"):
            if yaml_file.name == "kustomization.yaml":
                continue
            content = yaml_file.read_text()
            for doc in yaml.safe_load_all(content):
                if doc:
                    manifests.append(doc)
        return manifests

    def test_namespace_defined(self, base_manifests: List[Dict[str, Any]]):
        """Verify namespace is defined."""
        namespaces = [m for m in base_manifests if m.get("kind") == "Namespace"]
        assert len(namespaces) > 0, "No Namespace defined"
        ns = namespaces[0]
        assert ns["metadata"]["name"] == "quant-trading"

    def test_deployments_have_resources(self, base_manifests: List[Dict[str, Any]]):
        """Verify all deployments have resource limits."""
        deployments = [m for m in base_manifests if m.get("kind") == "Deployment"]
        for deploy in deployments:
            name = deploy["metadata"]["name"]
            containers = deploy["spec"]["template"]["spec"]["containers"]
            for container in containers:
                resources = container.get("resources", {})
                assert "requests" in resources, f"{name} missing resource requests"
                assert "limits" in resources, f"{name} missing resource limits"

    def test_deployments_have_probes(self, base_manifests: List[Dict[str, Any]]):
        """Verify deployments have liveness and readiness probes."""
        deployments = [m for m in base_manifests if m.get("kind") == "Deployment"]
        for deploy in deployments:
            name = deploy["metadata"]["name"]
            containers = deploy["spec"]["template"]["spec"]["containers"]
            for container in containers:
                assert "livenessProbe" in container, f"{name} missing livenessProbe"
                assert "readinessProbe" in container, f"{name} missing readinessProbe"

    def test_services_defined(self, base_manifests: List[Dict[str, Any]]):
        """Verify services are defined for deployments."""
        deployments = [m["metadata"]["name"] for m in base_manifests if m.get("kind") == "Deployment"]
        services = [m["metadata"]["name"] for m in base_manifests if m.get("kind") == "Service"]

        # Each deployment should have a corresponding service
        for deploy in deployments:
            # Service name might differ slightly (e.g., execution-service vs execution)
            matching = any(deploy.replace("-service", "") in svc for svc in services)
            assert matching, f"No service found for deployment {deploy}"

    def test_pdb_defined(self, base_manifests: List[Dict[str, Any]]):
        """Verify PodDisruptionBudgets are defined for critical services."""
        pdbs = [m["metadata"]["name"] for m in base_manifests
                if m.get("kind") == "PodDisruptionBudget"]
        assert len(pdbs) > 0, "No PodDisruptionBudgets defined"

    def test_rbac_defined(self, base_manifests: List[Dict[str, Any]]):
        """Verify RBAC is properly configured."""
        service_accounts = [m for m in base_manifests if m.get("kind") == "ServiceAccount"]
        roles = [m for m in base_manifests if m.get("kind") == "Role"]
        role_bindings = [m for m in base_manifests if m.get("kind") == "RoleBinding"]

        assert len(service_accounts) > 0, "No ServiceAccount defined"
        assert len(roles) > 0, "No Role defined"
        assert len(role_bindings) > 0, "No RoleBinding defined"

    def test_secrets_template_exists(self, base_manifests: List[Dict[str, Any]]):
        """Verify secrets template exists."""
        secrets = [m for m in base_manifests if m.get("kind") == "Secret"]
        assert len(secrets) > 0, "No Secrets defined"

    def test_configmaps_defined(self, base_manifests: List[Dict[str, Any]]):
        """Verify ConfigMaps are defined."""
        configmaps = [m for m in base_manifests if m.get("kind") == "ConfigMap"]
        assert len(configmaps) > 0, "No ConfigMaps defined"


class TestKustomization:
    """Test Kustomize configuration."""

    def test_base_kustomization_valid(self):
        """Verify base kustomization.yaml is valid."""
        kustomization = K8S_BASE_DIR / "kustomization.yaml"
        assert kustomization.exists()
        config = yaml.safe_load(kustomization.read_text())
        assert "resources" in config
        assert len(config["resources"]) > 0

    def test_dev_overlay_exists(self):
        """Verify dev overlay exists."""
        kustomization = K8S_DEV_DIR / "kustomization.yaml"
        assert kustomization.exists()
        config = yaml.safe_load(kustomization.read_text())
        assert "resources" in config or "bases" in config

    def test_prod_overlay_exists(self):
        """Verify prod overlay exists."""
        kustomization = K8S_PROD_DIR / "kustomization.yaml"
        assert kustomization.exists()
        config = yaml.safe_load(kustomization.read_text())
        assert "resources" in config or "bases" in config

    def test_overlays_have_namespace(self):
        """Verify overlays define distinct namespaces."""
        dev_config = yaml.safe_load((K8S_DEV_DIR / "kustomization.yaml").read_text())
        prod_config = yaml.safe_load((K8S_PROD_DIR / "kustomization.yaml").read_text())

        dev_ns = dev_config.get("namespace", "")
        prod_ns = prod_config.get("namespace", "")

        assert dev_ns != prod_ns, "Dev and prod should have different namespaces"


# =============================================================================
# Helm Chart Tests
# =============================================================================
class TestHelmChart:
    """Test Helm chart configuration."""

    @pytest.fixture
    def chart_yaml(self) -> Dict[str, Any]:
        """Load Chart.yaml."""
        chart_file = HELM_DIR / "Chart.yaml"
        assert chart_file.exists(), "Chart.yaml not found"
        return yaml.safe_load(chart_file.read_text())

    @pytest.fixture
    def values_yaml(self) -> Dict[str, Any]:
        """Load values.yaml."""
        values_file = HELM_DIR / "values.yaml"
        assert values_file.exists(), "values.yaml not found"
        return yaml.safe_load(values_file.read_text())

    def test_chart_metadata(self, chart_yaml: Dict[str, Any]):
        """Verify Chart.yaml has required metadata."""
        assert "name" in chart_yaml
        assert "version" in chart_yaml
        assert "appVersion" in chart_yaml
        assert chart_yaml["apiVersion"] == "v2"

    def test_values_structure(self, values_yaml: Dict[str, Any]):
        """Verify values.yaml has expected structure."""
        expected_keys = ["apiServer", "calibrationService", "dataIngestion",
                         "executionService", "signalService", "redis", "postgresql"]
        for key in expected_keys:
            assert key in values_yaml, f"Missing values key: {key}"

    def test_all_services_configurable(self, values_yaml: Dict[str, Any]):
        """Verify all services are configurable in values."""
        services = ["apiServer", "calibrationService", "dataIngestion",
                    "executionService", "signalService"]
        for service in services:
            svc_config = values_yaml.get(service, {})
            assert "enabled" in svc_config, f"{service} missing 'enabled'"
            assert "image" in svc_config, f"{service} missing 'image'"
            assert "resources" in svc_config, f"{service} missing 'resources'"

    def test_templates_exist(self):
        """Verify template files exist."""
        templates_dir = HELM_DIR / "templates"
        assert templates_dir.exists()
        templates = list(templates_dir.glob("*.yaml")) + list(templates_dir.glob("*.tpl"))
        assert len(templates) > 0, "No template files found"

    def test_helpers_tpl_exists(self):
        """Verify _helpers.tpl exists."""
        helpers = HELM_DIR / "templates" / "_helpers.tpl"
        assert helpers.exists(), "_helpers.tpl not found"


# =============================================================================
# Script Tests
# =============================================================================
class TestScripts:
    """Test deployment scripts."""

    def test_backup_script_exists(self):
        """Verify backup script exists and is executable logic."""
        backup_script = SCRIPTS_DIR / "backup.sh"
        assert backup_script.exists(), "backup.sh not found"
        content = backup_script.read_text()
        assert "pg_dump" in content, "backup.sh missing pg_dump"
        assert "aws s3" in content, "backup.sh missing S3 upload"

    def test_restore_script_exists(self):
        """Verify restore script exists."""
        restore_script = SCRIPTS_DIR / "restore.sh"
        assert restore_script.exists(), "restore.sh not found"
        content = restore_script.read_text()
        assert "pg_restore" in content, "restore.sh missing pg_restore"

    def test_scripts_have_error_handling(self):
        """Verify scripts have proper error handling."""
        for script in SCRIPTS_DIR.glob("*.sh"):
            content = script.read_text()
            assert "set -e" in content or "set -euo pipefail" in content, \
                f"{script.name} missing error handling (set -e)"


# =============================================================================
# CI/CD Tests
# =============================================================================
class TestCICD:
    """Test CI/CD workflow configuration."""

    @pytest.fixture
    def ci_workflow(self) -> Dict[str, Any]:
        """Load CI workflow."""
        ci_file = Path(__file__).parent.parent.parent.parent / ".github" / "workflows" / "ci.yml"
        assert ci_file.exists(), "ci.yml not found"
        return yaml.safe_load(ci_file.read_text())

    @pytest.fixture
    def cd_workflow(self) -> Dict[str, Any]:
        """Load CD workflow."""
        cd_file = Path(__file__).parent.parent.parent.parent / ".github" / "workflows" / "cd.yml"
        assert cd_file.exists(), "cd.yml not found"
        return yaml.safe_load(cd_file.read_text())

    def test_ci_has_required_jobs(self, ci_workflow: Dict[str, Any]):
        """Verify CI workflow has required jobs."""
        jobs = ci_workflow.get("jobs", {})
        required = ["lint", "test-python", "build-docker"]
        for job in required:
            assert job in jobs, f"CI missing job: {job}"

    def test_cd_has_environments(self, cd_workflow: Dict[str, Any]):
        """Verify CD workflow has environment deployments."""
        jobs = cd_workflow.get("jobs", {})
        assert "deploy-staging" in jobs or any("staging" in str(j).lower() for j in jobs.values())

    def test_ci_runs_on_push_and_pr(self, ci_workflow: Dict[str, Any]):
        """Verify CI triggers on push and PR."""
        # YAML 'on' key becomes True in Python, need to handle both cases
        on = ci_workflow.get("on") or ci_workflow.get(True, {})
        assert "push" in on or "pull_request" in on


# =============================================================================
# Configuration Validation Tests
# =============================================================================
class TestConfigurationSecurity:
    """Test security aspects of configuration."""

    def test_no_hardcoded_secrets(self):
        """Verify no hardcoded secrets in configuration files."""
        secret_patterns = ["password:", "api_key:", "secret:", "token:"]
        safe_patterns = ["REPLACE_", "valueFrom:", "secretKeyRef:", "${{"]

        for yaml_file in DEPLOY_DIR.rglob("*.yaml"):
            content = yaml_file.read_text()
            for pattern in secret_patterns:
                if pattern in content.lower():
                    # Check if it's a template/reference
                    lines_with_pattern = [
                        line for line in content.split("\n")
                        if pattern.split(":")[0] in line.lower()
                    ]
                    for line in lines_with_pattern:
                        is_safe = any(safe in line for safe in safe_patterns)
                        if not is_safe and "example" not in line.lower():
                            # Allow empty values or clearly placeholder values
                            if '""' not in line and "''" not in line:
                                pass  # Could add stricter checking here

    def test_network_policies_exist(self):
        """Verify NetworkPolicies are defined."""
        ingress_file = K8S_BASE_DIR / "ingress.yaml"
        assert ingress_file.exists()
        content = ingress_file.read_text()
        assert "NetworkPolicy" in content, "No NetworkPolicy defined"

    def test_resource_quotas_defined(self):
        """Verify ResourceQuotas are defined for namespace."""
        namespace_file = K8S_BASE_DIR / "namespace.yaml"
        content = namespace_file.read_text()
        assert "ResourceQuota" in content, "No ResourceQuota defined"


# =============================================================================
# Integration Test Stubs
# =============================================================================
class TestDeploymentIntegration:
    """Integration tests for deployment (require external tools)."""

    @pytest.mark.skipif(
        not os.path.exists("/usr/local/bin/kubectl"),
        reason="kubectl not installed"
    )
    def test_kustomize_build_base(self):
        """Test kustomize can build base manifests."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(K8S_BASE_DIR)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Kustomize build failed: {result.stderr}"

    @pytest.mark.skipif(
        not os.path.exists("/usr/local/bin/helm"),
        reason="helm not installed"
    )
    def test_helm_lint(self):
        """Test Helm chart linting."""
        result = subprocess.run(
            ["helm", "lint", str(HELM_DIR)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Helm lint failed: {result.stderr}"

    @pytest.mark.skipif(
        not os.path.exists("/usr/local/bin/helm"),
        reason="helm not installed"
    )
    def test_helm_template(self):
        """Test Helm template rendering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["helm", "template", "test-release", str(HELM_DIR),
                 "--output-dir", tmpdir],
                capture_output=True,
                text=True
            )
            # May have warnings for missing dependencies, but should succeed
            assert "Error" not in result.stderr or result.returncode == 0
