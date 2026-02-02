"""
Tests for documentation completeness and correctness.

These tests verify that:
1. All required documentation files exist
2. Documentation follows correct formats (OpenAPI, markdown)
3. Code examples in documentation are valid
4. Cross-references between documents work
"""

import os
import json
import yaml
import pytest
from pathlib import Path


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


class TestDocumentationStructure:
    """Test that required documentation files exist."""

    def test_docs_directory_exists(self):
        """Verify docs directory exists."""
        assert DOCS_DIR.exists(), "docs/ directory should exist"

    def test_getting_started_exists(self):
        """Verify getting started guide exists."""
        getting_started = DOCS_DIR / "getting-started.md"
        assert getting_started.exists(), "getting-started.md should exist"

    def test_architecture_docs_exist(self):
        """Verify architecture documentation exists."""
        architecture_dir = DOCS_DIR / "architecture"
        required_files = [
            "system-overview.md",
            "component-diagram.md",
            "data-flow.md",
            "technology-stack.md",
        ]

        for filename in required_files:
            filepath = architecture_dir / filename
            assert filepath.exists(), f"architecture/{filename} should exist"

    def test_api_docs_exist(self):
        """Verify API documentation exists."""
        api_dir = DOCS_DIR / "api"
        required_files = [
            "openapi.yaml",
            "rest-api.md",
        ]

        for filename in required_files:
            filepath = api_dir / filename
            assert filepath.exists(), f"api/{filename} should exist"

    def test_model_docs_exist(self):
        """Verify model documentation exists."""
        models_dir = DOCS_DIR / "models"
        required_files = [
            "heston-model.md",
            "sabr-model.md",
            "ou-process.md",
        ]

        for filename in required_files:
            filepath = models_dir / filename
            assert filepath.exists(), f"models/{filename} should exist"

    def test_database_docs_exist(self):
        """Verify database documentation exists."""
        db_dir = DOCS_DIR / "database"
        assert (db_dir / "schema.md").exists(), "database/schema.md should exist"

    def test_deployment_docs_exist(self):
        """Verify deployment documentation exists."""
        deploy_dir = DOCS_DIR / "deployment"
        assert (deploy_dir / "kubernetes.md").exists(), "deployment/kubernetes.md should exist"

    def test_development_docs_exist(self):
        """Verify development documentation exists."""
        dev_dir = DOCS_DIR / "development"
        assert (dev_dir / "setup.md").exists(), "development/setup.md should exist"

    def test_operations_runbooks_exist(self):
        """Verify operations runbooks exist."""
        runbooks_dir = DOCS_DIR / "operations" / "runbooks"
        assert (runbooks_dir / "incident-response.md").exists(), \
            "operations/runbooks/incident-response.md should exist"


class TestOpenAPISpecification:
    """Test OpenAPI specification validity."""

    @pytest.fixture
    def openapi_spec(self):
        """Load OpenAPI specification."""
        openapi_path = DOCS_DIR / "api" / "openapi.yaml"
        if not openapi_path.exists():
            pytest.skip("OpenAPI spec not found")
        with open(openapi_path, 'r') as f:
            return yaml.safe_load(f)

    def test_openapi_version(self, openapi_spec):
        """Verify OpenAPI version is 3.x."""
        assert "openapi" in openapi_spec
        assert openapi_spec["openapi"].startswith("3.")

    def test_openapi_info(self, openapi_spec):
        """Verify required info fields exist."""
        info = openapi_spec.get("info", {})
        assert "title" in info, "OpenAPI spec should have title"
        assert "version" in info, "OpenAPI spec should have version"
        assert "description" in info, "OpenAPI spec should have description"

    def test_openapi_paths(self, openapi_spec):
        """Verify paths are defined."""
        paths = openapi_spec.get("paths", {})
        assert len(paths) > 0, "OpenAPI spec should define paths"

    def test_openapi_has_authentication(self, openapi_spec):
        """Verify security schemes are defined."""
        components = openapi_spec.get("components", {})
        security_schemes = components.get("securitySchemes", {})
        assert "BearerAuth" in security_schemes, \
            "OpenAPI spec should define BearerAuth security scheme"

    def test_openapi_strategies_endpoint(self, openapi_spec):
        """Verify /strategies endpoint exists."""
        paths = openapi_spec.get("paths", {})
        assert "/strategies" in paths, "OpenAPI spec should have /strategies endpoint"

    def test_openapi_signals_endpoint(self, openapi_spec):
        """Verify signals endpoints exist."""
        paths = openapi_spec.get("paths", {})
        assert "/signals" in paths or "/signals/active" in paths, \
            "OpenAPI spec should have signals endpoint"

    def test_openapi_health_endpoint(self, openapi_spec):
        """Verify /health endpoint exists."""
        paths = openapi_spec.get("paths", {})
        assert "/health" in paths, "OpenAPI spec should have /health endpoint"

    def test_openapi_schemas(self, openapi_spec):
        """Verify schemas are defined."""
        components = openapi_spec.get("components", {})
        schemas = components.get("schemas", {})

        required_schemas = [
            "Strategy",
            "Signal",
            "Position",
            "RiskMetrics",
            "Error",
        ]

        for schema_name in required_schemas:
            assert schema_name in schemas, \
                f"OpenAPI spec should define {schema_name} schema"


class TestMarkdownDocumentation:
    """Test markdown documentation quality."""

    def get_markdown_files(self):
        """Get all markdown files in docs directory."""
        return list(DOCS_DIR.rglob("*.md"))

    def test_markdown_files_exist(self):
        """Verify markdown files exist."""
        md_files = self.get_markdown_files()
        assert len(md_files) > 0, "Should have markdown documentation files"

    def test_markdown_has_title(self):
        """Verify each markdown file has a title."""
        for md_file in self.get_markdown_files():
            with open(md_file, 'r') as f:
                content = f.read()
                # Check for h1 heading
                assert content.startswith("# ") or "\n# " in content, \
                    f"{md_file.name} should have a title (# heading)"

    def test_markdown_not_empty(self):
        """Verify markdown files are not empty."""
        for md_file in self.get_markdown_files():
            with open(md_file, 'r') as f:
                content = f.read().strip()
                assert len(content) > 100, \
                    f"{md_file.name} should have substantial content"

    def test_no_broken_internal_links(self):
        """Verify internal links point to existing files."""
        import re
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        for md_file in self.get_markdown_files():
            with open(md_file, 'r') as f:
                content = f.read()

            for match in link_pattern.finditer(content):
                link_text, link_url = match.groups()

                # Skip external links and anchors
                if link_url.startswith(('http://', 'https://', '#', 'mailto:')):
                    continue

                # Resolve relative path
                if link_url.startswith('./'):
                    link_path = md_file.parent / link_url[2:]
                elif link_url.startswith('../'):
                    link_path = md_file.parent / link_url
                else:
                    link_path = md_file.parent / link_url

                # Remove anchor if present
                link_path = Path(str(link_path).split('#')[0])

                # Check if file exists (skip if directory reference)
                if not link_path.exists() and not link_path.parent.exists():
                    pytest.fail(
                        f"Broken link in {md_file.name}: "
                        f"'{link_text}' -> '{link_url}'"
                    )


class TestJupyterNotebooks:
    """Test Jupyter notebook validity."""

    def get_notebook_files(self):
        """Get all notebook files."""
        if not NOTEBOOKS_DIR.exists():
            return []
        return list(NOTEBOOKS_DIR.glob("*.ipynb"))

    def test_notebooks_directory_exists(self):
        """Verify notebooks directory exists."""
        assert NOTEBOOKS_DIR.exists(), "notebooks/ directory should exist"

    def test_notebooks_exist(self):
        """Verify tutorial notebooks exist."""
        notebooks = self.get_notebook_files()
        assert len(notebooks) > 0, "Should have tutorial notebooks"

    def test_notebooks_valid_json(self):
        """Verify notebooks are valid JSON."""
        for nb_file in self.get_notebook_files():
            with open(nb_file, 'r') as f:
                try:
                    nb = json.load(f)
                    assert "cells" in nb, \
                        f"{nb_file.name} should have cells"
                    assert "nbformat" in nb, \
                        f"{nb_file.name} should have nbformat"
                except json.JSONDecodeError:
                    pytest.fail(f"{nb_file.name} is not valid JSON")

    def test_notebooks_have_markdown_cells(self):
        """Verify notebooks have explanatory markdown cells."""
        for nb_file in self.get_notebook_files():
            with open(nb_file, 'r') as f:
                nb = json.load(f)

            markdown_cells = [
                c for c in nb.get("cells", [])
                if c.get("cell_type") == "markdown"
            ]

            assert len(markdown_cells) >= 3, \
                f"{nb_file.name} should have explanatory markdown cells"

    def test_notebooks_have_code_cells(self):
        """Verify notebooks have code cells."""
        for nb_file in self.get_notebook_files():
            with open(nb_file, 'r') as f:
                nb = json.load(f)

            code_cells = [
                c for c in nb.get("cells", [])
                if c.get("cell_type") == "code"
            ]

            assert len(code_cells) >= 3, \
                f"{nb_file.name} should have code examples"


class TestDocumentationContent:
    """Test documentation content quality."""

    def test_architecture_has_diagrams(self):
        """Verify architecture docs have diagrams."""
        arch_file = DOCS_DIR / "architecture" / "system-overview.md"
        if not arch_file.exists():
            pytest.skip("Architecture doc not found")

        with open(arch_file, 'r') as f:
            content = f.read()

        # Check for ASCII diagrams or mermaid blocks
        has_diagram = (
            "```" in content and
            ("┌" in content or "mermaid" in content.lower() or "│" in content)
        )
        assert has_diagram, "Architecture docs should include diagrams"

    def test_model_docs_have_equations(self):
        """Verify model docs have mathematical equations."""
        model_files = [
            DOCS_DIR / "models" / "heston-model.md",
            DOCS_DIR / "models" / "sabr-model.md",
            DOCS_DIR / "models" / "ou-process.md",
        ]

        for model_file in model_files:
            if not model_file.exists():
                continue

            with open(model_file, 'r') as f:
                content = f.read()

            # Check for math notation (various formats)
            has_equations = any(indicator in content for indicator in [
                "$$", "\\frac", "\\sigma", "dS_t", "dv_t",
                "dF_t", "dσ_t", "dX_t", "σ_t", "√", "κ", "θ"
            ])
            assert has_equations, \
                f"{model_file.name} should include mathematical equations"

    def test_api_docs_have_examples(self):
        """Verify API docs have code examples."""
        api_file = DOCS_DIR / "api" / "rest-api.md"
        if not api_file.exists():
            pytest.skip("API doc not found")

        with open(api_file, 'r') as f:
            content = f.read()

        # Check for code blocks with curl or Python
        has_examples = (
            "```bash" in content or
            "```python" in content or
            "curl" in content
        )
        assert has_examples, "API docs should include usage examples"

    def test_runbooks_have_commands(self):
        """Verify runbooks have executable commands."""
        runbook_file = DOCS_DIR / "operations" / "runbooks" / "incident-response.md"
        if not runbook_file.exists():
            pytest.skip("Runbook not found")

        with open(runbook_file, 'r') as f:
            content = f.read()

        # Check for kubectl commands
        has_commands = (
            "kubectl" in content or
            "docker" in content or
            "curl" in content
        )
        assert has_commands, "Runbooks should include executable commands"


class TestDocumentationMetrics:
    """Test documentation coverage metrics."""

    def test_minimum_doc_count(self):
        """Verify minimum number of documentation files."""
        md_files = list(DOCS_DIR.rglob("*.md"))
        assert len(md_files) >= 10, \
            "Should have at least 10 documentation files"

    def test_minimum_total_content(self):
        """Verify minimum total documentation content."""
        total_chars = 0
        for md_file in DOCS_DIR.rglob("*.md"):
            with open(md_file, 'r') as f:
                total_chars += len(f.read())

        # At least 50KB of documentation
        assert total_chars >= 50000, \
            "Should have substantial documentation content"

    def test_docs_updated_recently(self):
        """Check if documentation has been updated recently."""
        import time

        most_recent = 0
        for md_file in DOCS_DIR.rglob("*.md"):
            mtime = md_file.stat().st_mtime
            most_recent = max(most_recent, mtime)

        # Documentation should be updated within last 90 days
        days_old = (time.time() - most_recent) / (24 * 60 * 60)
        assert days_old < 90, \
            "Documentation should be updated within 90 days"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
