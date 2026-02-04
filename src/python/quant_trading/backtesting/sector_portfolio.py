"""
Sector-Based Portfolio with Confidence-Weighted Position Sizing.

Implements a diversified portfolio across multiple sectors with:
- Sector-specific strategy assignments based on characteristics
- Confidence calculation using OU fitting, volatility analysis, momentum strength
- Dynamic position sizing based on signal confidence
- Risk management through sector exposure limits

Sectors:
- Technology (high growth, momentum-driven)
- Financials (interest rate sensitive, mean-reverting)
- Healthcare (defensive, steady growth)
- Consumer Discretionary (cyclical, momentum)
- Consumer Staples (defensive, low volatility)
- Energy (commodity-driven, high volatility)
- Industrials (cyclical, economic sensitive)
- Materials (commodity-linked)
- Utilities (defensive, income)
- Real Estate (interest rate sensitive)
- Communication Services (growth/value mix)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .sector_optimizer import SectorOptimizationResults
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class Sector(Enum):
    """Market sectors for portfolio construction."""
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"
    HEALTHCARE = "healthcare"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    ENERGY = "energy"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMUNICATION = "communication"
    ETF_INDEX = "etf_index"
    ETF_SECTOR = "etf_sector"


# Comprehensive stock universe by sector (400+ stocks)
SECTOR_STOCKS: Dict[Sector, List[str]] = {
    Sector.TECHNOLOGY: [
        # Mega-cap tech
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ACN", "ADBE", "CRM", "AMD",
        # Semiconductors
        "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "NXPI",
        "ADI", "MPWR", "SWKS", "QRVO", "TER", "ENTG", "MKSI", "CRUS", "SYNA", "SMTC",
        # Software
        "NOW", "INTU", "PANW", "SNPS", "CDNS", "FTNT", "WDAY", "TEAM", "CRWD", "ZS",
        "DDOG", "NET", "MDB", "SNOW", "PLTR", "HUBS", "VEEV", "ANSS", "PAYC", "PCTY",
        "BILL", "DOCU", "ZEN", "OKTA", "SPLK", "ESTC", "PATH", "DOCN", "GTLB",
        # Hardware & IT Services
        "IBM", "HPQ", "HPE", "DELL", "NTAP", "WDC", "STX", "PSTG", "JNPR", "ANET",
        "FFIV", "AKAM", "EPAM", "GLOB", "CTSH", "INFY", "WIT",
    ],
    Sector.FINANCIALS: [
        # Big banks
        "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
        # Insurance
        "BRK-B", "AIG", "MET", "PRU", "ALL", "TRV", "AFL", "PGR", "CB", "HIG",
        "CINF", "L", "RE", "WRB", "AIZ", "LNC", "UNM", "GL", "CNO", "PRI",
        # Asset managers & exchanges
        "BLK", "SCHW", "CME", "ICE", "SPGI", "MCO", "MSCI", "FIS", "FISV", "AXP",
        "NDAQ", "CBOE", "MKTX", "VIRT", "LPLA", "RJF", "SEIC", "AMG", "BEN", "IVZ",
        # Regional banks
        "FITB", "RF", "HBAN", "KEY", "CFG", "MTB", "ZION", "CMA", "FHN", "SNV",
        "BOKF", "PNFP", "UMBF", "WTFC", "CBSH", "FNB", "PACW", "WAL", "FFIN",
        # Fintech
        "V", "MA", "PYPL", "SQ", "AFRM", "UPST", "SOFI", "HOOD", "COIN",
    ],
    Sector.HEALTHCARE: [
        # Pharma
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AZN", "NVO", "SNY", "GSK", "NVS", "TAK", "VTRS", "TEVA", "ZTS", "ELAN",
        # Biotech
        "AMGN", "GILD", "VRTX", "REGN", "BIIB", "MRNA", "ILMN", "SGEN", "ALNY", "BMRN",
        "EXEL", "INCY", "SRPT", "IONS", "RARE", "HALO", "UTHR", "NBIX", "PCVX", "ACAD",
        # Medical devices
        "MDT", "SYK", "BSX", "EW", "ISRG", "ZBH", "BDX", "HOLX", "ALGN", "DXCM",
        "IDXX", "TFX", "COO", "RMD", "BAX", "PODD", "NVST", "INSP", "SILK",
        # Health services
        "CVS", "CI", "ELV", "HCA", "HUM", "CNC", "MCK", "CAH", "ABC", "HSIC",
        "DVA", "THC", "UHS", "ACHC", "SGRY", "ENSG", "AMED", "LHCG", "PNTG",
    ],
    Sector.CONSUMER_DISCRETIONARY: [
        # Retail
        "AMZN", "HD", "LOW", "TJX", "NKE", "SBUX", "TGT", "ROST", "ORLY", "AZO",
        "BURL", "FIVE", "WSM", "RH", "W", "ETSY", "CHWY", "CVNA", "KMX", "AN",
        # Auto
        "TSLA", "GM", "F", "RIVN", "LCID", "STLA", "HMC", "TM", "APTV", "LEA",
        "BWA", "ALV", "VC", "GNTX", "DAN", "MGA", "THRM", "MOD", "CWH",
        # Travel & leisure
        "MCD", "CMG", "DRI", "YUM", "SBUX", "QSR", "WING", "TXRH", "CAKE", "EAT",
        "MAR", "HLT", "H", "WH", "CHH", "IHG", "ABNB", "BKNG", "EXPE", "TRIP",
        "RCL", "CCL", "NCLH", "LUV", "DAL", "UAL", "AAL", "JBLU", "ALK", "SAVE",
        # Other consumer
        "LULU", "DG", "DLTR", "BBY", "ULTA", "GPS", "ANF", "AEO", "URBN", "EXPR",
        "DECK", "CROX", "SKX", "FL", "GOOS", "VFC", "PVH", "RL", "TPR", "CPRI",
    ],
    Sector.CONSUMER_STAPLES: [
        # Food & beverage
        "PG", "KO", "PEP", "COST", "WMT", "MDLZ", "MO", "PM", "CL", "KMB",
        "KDP", "STZ", "BF-B", "DEO", "TAP", "SAM", "MNST",
        # Grocery & household
        "KR", "SYY", "HSY", "K", "GIS", "CAG", "CPB", "SJM", "HRL", "MKC",
        "TSN", "PPC", "INGR", "DAR", "POST", "THS", "BGS", "FLO",
        # Personal products
        "EL", "CHD", "CLX", "CL", "COTY", "EPC", "HELE", "SPB",
        # Discount & grocery retail
        "DG", "DLTR", "BJ", "GO", "SFM", "NGVC", "UNFI", "SPTN",
    ],
    Sector.ENERGY: [
        # Oil majors
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HES",
        "DVN", "FANG", "MRO", "APA", "MGY", "CTRA", "MTDR", "PR", "SM", "OVV",
        # Oil services
        "HAL", "BKR", "NOV", "FTI", "CHX", "WHD", "WFRD", "RES", "HP", "PTEN",
        "SWN", "RRC", "AR", "EQT", "CNX", "COG",
        # Natural gas & pipelines
        "KMI", "WMB", "OKE", "ET", "EPD", "TRGP", "LNG", "DTM", "ENLC", "PAA",
        "MPLX", "PSXP", "CEQP", "HESM", "GEL", "NS",
        # Refiners
        "DK", "PBF", "PARR", "CVI", "HFC",
    ],
    Sector.INDUSTRIALS: [
        # Aerospace & defense
        "RTX", "BA", "LMT", "NOC", "GD", "LHX", "TDG", "HWM", "HII", "TXT",
        "SPR", "HEI", "MOG-A", "AXON", "KTOS", "MRCY", "AVAV", "RKLB",
        # Industrial conglomerates
        "HON", "GE", "MMM", "CAT", "DE", "EMR", "ETN", "ROK", "PH", "ITW",
        "CMI", "DOV", "IR", "XYL", "GGG", "GNRC", "FLS", "FELE", "RBC",
        # Transportation
        "UNP", "UPS", "FDX", "CSX", "NSC", "DAL", "UAL", "LUV", "AAL", "JBLU",
        "CHRW", "EXPD", "XPO", "ODFL", "SAIA", "JBHT", "LSTR", "KNX", "WERN",
        # Construction & engineering
        "JCI", "CARR", "OTIS", "IR", "SWK", "LII", "TT", "WSO", "TREX", "AZEK",
        "MAS", "FND", "BLD", "OC", "BLDR", "UFPI", "WMS", "CSL", "VMC", "MLM",
        # Machinery
        "PCAR", "WAB", "GWW", "FAST", "MSM", "AIT", "WCC", "CNH", "AGCO", "TTC",
    ],
    Sector.MATERIALS: [
        # Chemicals
        "LIN", "APD", "SHW", "ECL", "DD", "DOW", "PPG", "NEM", "FCX", "LYB",
        "EMN", "CE", "HUN", "WLK", "OLN", "CC", "OLIN", "KWR", "CBT", "ASH",
        # Metals & mining
        "NUE", "STLD", "CF", "MOS", "ALB", "FMC", "AA", "CENX", "CLF", "X",
        "RS", "ATI", "CMC", "SCCO", "TECK", "RIO", "BHP", "VALE", "MT",
        # Paper & packaging
        "IP", "PKG", "AVY", "SEE", "BLL", "CCK", "SLGN", "SON", "GPK", "BERY",
        # Construction materials
        "VMC", "MLM", "EXP", "SUM", "USLM", "CRH", "JHX",
    ],
    Sector.UTILITIES: [
        # Electric utilities
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL", "PEG", "ED", "EXC",
        "PCG", "EIX", "FE", "ETR", "PPL", "CMS", "AES", "NRG", "VST", "OGE",
        # Multi-utilities
        "WEC", "ES", "AWK", "ATO", "NI", "DTE", "CNP", "LNT", "EVRG", "PNW",
        # Gas utilities
        "SWX", "NJR", "OGS", "SR", "SPH", "NFG",
        # Water utilities
        "AWK", "WTR", "AWR", "WTRG", "CWT", "MSEX", "SJW", "YORW",
        # Renewable energy
        "BEP", "CWEN", "NEP", "AY", "RUN", "NOVA", "ENPH", "SEDG",
    ],
    Sector.REAL_ESTATE: [
        # REITs - diverse
        "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
        "EQR", "VTR", "ARE", "MAA", "UDR", "ESS", "INVH", "SUI", "ELS", "CPT",
        # Office REITs
        "BXP", "VNO", "SLG", "KRC", "CUZ", "HIW", "DEI", "PGRE", "CLI", "PDM",
        # Retail REITs
        "SPG", "MAC", "KIM", "REG", "FRT", "BRX", "ROIC", "AKR", "SITC", "UE",
        # Industrial REITs
        "PLD", "DRE", "EGP", "FR", "STAG", "TRNO", "PLYM", "GTY",
        # Data center & tower
        "AMT", "CCI", "SBAC", "DLR", "EQIX", "CONE", "QTS",
        # Healthcare REITs
        "WELL", "VTR", "PEAK", "HR", "OHI", "NHI", "LTC", "CTRE", "SBRA",
    ],
    Sector.COMMUNICATION: [
        # Telecom
        "T", "VZ", "TMUS", "CMCSA", "CHTR", "LBRDK", "LBRDA", "FYBR", "LUMN", "FTR",
        "ATUS", "CABO", "TDS", "USM", "SHEN",
        # Media & entertainment
        "DIS", "NFLX", "WBD", "PARA", "FOX", "FOXA", "NWS", "NWSA", "NYT", "GCI",
        "VIAC", "DISCA", "DISCB", "DISCK", "AMC", "CNK", "IMAX", "LGF-A", "LGF-B",
        # Internet
        "GOOG", "GOOGL", "META", "SNAP", "PINS", "MTCH", "IAC", "BMBL", "ZG", "Z",
        "YELP", "TRIP", "ANGI", "CARS", "OPEN", "GRPN", "EVER",
        # Gaming
        "EA", "TTWO", "ATVI", "RBLX", "U", "ZNGA", "GLUU",
    ],
    Sector.ETF_INDEX: [
        # Broad market
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "RSP", "SPLG", "ITOT",
        "VV", "MGC", "OEF", "SCHX", "SPTM", "VXF", "SCHA", "IJH", "MDY", "VO",
        # International
        "EFA", "EEM", "VEA", "VWO", "IEFA", "IEMG", "ACWI", "ACWX", "VT", "VXUS",
        "FXI", "EWJ", "EWZ", "EWG", "EWU", "EWY", "EWT", "INDA", "VNM", "FM",
        # Bonds
        "TLT", "IEF", "LQD", "HYG", "AGG", "BND", "VCIT", "VCSH", "VGIT", "VGSH",
        "JNK", "BNDX", "EMB", "MUB", "TIP", "GOVT", "SHY", "IEI", "SCHZ",
    ],
    Sector.ETF_SECTOR: [
        # Sector ETFs
        "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE",
        "VGT", "VFH", "VHT", "VCR", "VDC", "VDE", "VIS", "VAW", "VPU", "VNQ",
        # Thematic
        "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ", "SOXX", "SMH", "XBI", "IBB", "HACK",
        "BOTZ", "ROBO", "SKYY", "WCLD", "FINX", "IPAY", "BLOK", "GNOM",
        # Commodity
        "GDX", "GDXJ", "SLV", "GLD", "IAU", "USO", "UNG", "DBA", "DBC", "PDBC",
        # Volatility & leverage
        "VXX", "UVXY", "SVXY", "TQQQ", "SQQQ", "SPXU", "UPRO", "TNA", "TZA",
    ],
}

# Optimal strategy assignments by sector
SECTOR_STRATEGIES: Dict[Sector, Dict] = {
    Sector.TECHNOLOGY: {
        "primary": "momentum",
        "params": {"lookback": 10, "threshold": 0.03},
        "secondary": "ma_crossover",
        "secondary_params": {"fast": 3, "slow": 10},
    },
    Sector.FINANCIALS: {
        "primary": "mean_reversion",
        "params": {"lookback": 15, "entry_threshold": 2.0, "exit_threshold": 0.5},
        "secondary": "ma_crossover",
        "secondary_params": {"fast": 5, "slow": 15},
    },
    Sector.HEALTHCARE: {
        "primary": "ma_crossover",
        "params": {"fast": 3, "slow": 12},
        "secondary": "momentum",
        "secondary_params": {"lookback": 15, "threshold": 0.02},
    },
    Sector.CONSUMER_DISCRETIONARY: {
        "primary": "momentum",
        "params": {"lookback": 10, "threshold": 0.025},
        "secondary": "ma_crossover",
        "secondary_params": {"fast": 3, "slow": 10},
    },
    Sector.CONSUMER_STAPLES: {
        "primary": "ma_crossover",
        "params": {"fast": 5, "slow": 15},
        "secondary": "rsi",
        "secondary_params": {"period": 14, "oversold": 35, "overbought": 65},
    },
    Sector.ENERGY: {
        "primary": "momentum",
        "params": {"lookback": 8, "threshold": 0.04},
        "secondary": "bollinger",
        "secondary_params": {"period": 15, "num_std": 2.0},
    },
    Sector.INDUSTRIALS: {
        "primary": "ma_crossover",
        "params": {"fast": 3, "slow": 12},
        "secondary": "momentum",
        "secondary_params": {"lookback": 12, "threshold": 0.025},
    },
    Sector.MATERIALS: {
        "primary": "momentum",
        "params": {"lookback": 10, "threshold": 0.035},
        "secondary": "mean_reversion",
        "secondary_params": {"lookback": 12, "entry_threshold": 1.8, "exit_threshold": 0.5},
    },
    Sector.UTILITIES: {
        "primary": "mean_reversion",
        "params": {"lookback": 20, "entry_threshold": 1.5, "exit_threshold": 0.3},
        "secondary": "rsi",
        "secondary_params": {"period": 14, "oversold": 30, "overbought": 70},
    },
    Sector.REAL_ESTATE: {
        "primary": "mean_reversion",
        "params": {"lookback": 15, "entry_threshold": 1.8, "exit_threshold": 0.5},
        "secondary": "ma_crossover",
        "secondary_params": {"fast": 5, "slow": 15},
    },
    Sector.COMMUNICATION: {
        "primary": "ma_crossover",
        "params": {"fast": 3, "slow": 10},
        "secondary": "momentum",
        "secondary_params": {"lookback": 10, "threshold": 0.03},
    },
    Sector.ETF_INDEX: {
        "primary": "momentum",
        "params": {"lookback": 10, "threshold": 0.02},
        "secondary": "ma_crossover",
        "secondary_params": {"fast": 5, "slow": 20},
    },
    Sector.ETF_SECTOR: {
        "primary": "momentum",
        "params": {"lookback": 8, "threshold": 0.025},
        "secondary": "ma_crossover",
        "secondary_params": {"fast": 3, "slow": 10},
    },
}

# Stock-to-sector mapping (auto-generated from SECTOR_STOCKS)
STOCK_TO_SECTOR: Dict[str, Sector] = {}
for sector, stocks in SECTOR_STOCKS.items():
    for stock in stocks:
        STOCK_TO_SECTOR[stock] = sector


def get_sector(symbol: str) -> Sector:
    """Get the sector for a given symbol."""
    return STOCK_TO_SECTOR.get(symbol.upper(), Sector.TECHNOLOGY)  # Default to tech


def get_sector_strategy(symbol: str) -> Dict:
    """Get the optimal strategy configuration for a symbol based on its sector."""
    sector = get_sector(symbol)
    strategy_config = SECTOR_STRATEGIES.get(sector, SECTOR_STRATEGIES[Sector.TECHNOLOGY])
    return {
        "type": strategy_config["primary"],
        "params": strategy_config["params"].copy(),
        "sector": sector.value,
    }


def get_all_stocks() -> List[str]:
    """Get all stocks in the universe."""
    all_stocks = []
    for stocks in SECTOR_STOCKS.values():
        all_stocks.extend(stocks)
    return list(set(all_stocks))


def get_stocks_by_sector(sector: Sector) -> List[str]:
    """Get all stocks in a given sector."""
    return SECTOR_STOCKS.get(sector, [])


@dataclass
class ConfidenceMetrics:
    """
    Confidence metrics for position sizing.

    Combines multiple signals to determine overall confidence:
    - Trend strength (momentum/MA alignment)
    - Mean-reversion quality (half-life, R-squared)
    - Volatility regime (realized vs implied)
    - Signal agreement (primary vs secondary strategy)
    - Sector-algorithm fitness (from optimization results)
    """

    symbol: str
    sector: Sector

    # Trend metrics
    momentum_strength: float = 0.0  # Normalized momentum score [-1, 1]
    trend_alignment: float = 0.0  # MA alignment score [0, 1]

    # Mean-reversion metrics (from OU fitting)
    half_life_days: float = float('inf')  # Days to revert 50%
    mean_reversion_score: float = 0.0  # 0-1 score based on half-life
    z_score: float = 0.0  # Current deviation from mean

    # Volatility metrics
    realized_volatility: float = 0.0  # 20-day realized vol
    volatility_percentile: float = 0.5  # Current vol vs historical

    # Signal quality
    signal_strength: float = 0.0  # Primary strategy signal strength
    strategy_agreement: float = 0.0  # Agreement between strategies

    # Sector-algorithm fitness (from optimization)
    sector_algorithm_fitness: float = 0.5  # 0-1 from sector optimization

    # Overall confidence
    confidence: float = 0.0  # Final confidence score [0, 1]

    def calculate_confidence(self, use_fitness: bool = True) -> float:
        """
        Calculate overall confidence score.

        When use_fitness=True (optimization results available):
        - Trend strength: 20%
        - Mean-reversion quality: 15%
        - Volatility regime: 15%
        - Signal strength: 20%
        - Strategy agreement: 10%
        - Sector-algorithm fitness: 20%

        When use_fitness=False (no optimization):
        - Trend strength: 25%
        - Mean-reversion quality: 20%
        - Volatility regime: 15%
        - Signal strength: 25%
        - Strategy agreement: 15%
        """
        # Normalize components to [0, 1]
        trend_score = (self.momentum_strength + 1) / 2  # Convert [-1,1] to [0,1]
        mr_score = self.mean_reversion_score
        vol_score = 1 - self.volatility_percentile  # Lower vol = higher confidence
        signal_score = self.signal_strength
        agreement_score = self.strategy_agreement
        fitness_score = self.sector_algorithm_fitness

        if use_fitness and self.sector_algorithm_fitness != 0.5:
            # Weights when optimization data is available
            self.confidence = (
                0.20 * trend_score +
                0.15 * mr_score +
                0.15 * vol_score +
                0.20 * signal_score +
                0.10 * agreement_score +
                0.20 * fitness_score
            )
        else:
            # Original weights without optimization
            self.confidence = (
                0.25 * trend_score +
                0.20 * mr_score +
                0.15 * vol_score +
                0.25 * signal_score +
                0.15 * agreement_score
            )

        return self.confidence


class ConfidenceCalculator:
    """
    Calculates confidence metrics for position sizing.

    Uses price history and optional C++ bindings for:
    - OU parameter fitting (mean-reversion detection)
    - Momentum calculation
    - Volatility analysis
    - Multi-strategy signal agreement
    - Sector-algorithm fitness (from optimization results)
    """

    def __init__(
        self,
        lookback_days: int = 60,
        optimization_results: Optional["SectorOptimizationResults"] = None,
    ):
        """
        Initialize confidence calculator.

        Args:
            lookback_days: Historical lookback for calculations
            optimization_results: Optional optimization results for fitness scoring
        """
        self.lookback_days = lookback_days
        self._ou_fitter = None
        self._cpp_available = False
        self._optimization_results = optimization_results

        # Try to load C++ bindings
        try:
            from ..calibration import OUFitter
            self._ou_fitter = OUFitter()
            logger.info("OU Fitter initialized for confidence calculation")
        except ImportError:
            logger.warning("OUFitter not available, using simplified confidence")

        # Try to load C++ extensions
        try:
            from ..cpp import is_available
            self._cpp_available = is_available()
            if self._cpp_available:
                logger.info("C++ extensions available for confidence calculation")
        except ImportError:
            pass

    def set_optimization_results(self, results: "SectorOptimizationResults") -> None:
        """Set or update optimization results for fitness scoring."""
        self._optimization_results = results
        logger.info("Optimization results set for confidence calculation")

    def calculate(
        self,
        symbol: str,
        prices: np.ndarray,
        signal_strength: float = 0.5,
        algorithm: Optional[str] = None,
    ) -> ConfidenceMetrics:
        """
        Calculate confidence metrics for a symbol.

        Args:
            symbol: Stock ticker
            prices: Price history array
            signal_strength: Strength from primary strategy
            algorithm: Algorithm being used (for fitness lookup)

        Returns:
            ConfidenceMetrics with all scores
        """
        sector = get_sector(symbol)
        metrics = ConfidenceMetrics(symbol=symbol, sector=sector)

        if len(prices) < 20:
            metrics.confidence = 0.3  # Low confidence for short history
            return metrics

        # Calculate momentum strength
        metrics.momentum_strength = self._calculate_momentum(prices)

        # Calculate trend alignment (MA crossover)
        metrics.trend_alignment = self._calculate_trend_alignment(prices)

        # Calculate mean-reversion metrics
        if self._ou_fitter and len(prices) >= 30:
            try:
                result = self._ou_fitter.fit(prices, dt=1/252, compute_boundaries=False)
                if result.success:
                    metrics.half_life_days = result.params.half_life
                    # Score based on half-life (5-30 days is ideal for swing trading)
                    if 5 <= metrics.half_life_days <= 30:
                        metrics.mean_reversion_score = 1.0 - abs(metrics.half_life_days - 15) / 15
                    elif metrics.half_life_days < 5:
                        metrics.mean_reversion_score = 0.5  # Too fast
                    else:
                        metrics.mean_reversion_score = max(0, 1.0 - (metrics.half_life_days - 30) / 60)

                    # Calculate z-score
                    mean = result.params.theta
                    std = result.params.stationary_std
                    if std > 0:
                        metrics.z_score = (prices[-1] - mean) / std
            except Exception as e:
                logger.debug(f"OU fitting failed for {symbol}: {e}")

        # Calculate volatility metrics
        metrics.realized_volatility = self._calculate_volatility(prices)
        metrics.volatility_percentile = self._calculate_vol_percentile(prices)

        # Set signal strength
        metrics.signal_strength = signal_strength

        # Calculate strategy agreement (simplified)
        metrics.strategy_agreement = self._calculate_strategy_agreement(
            metrics.momentum_strength,
            metrics.trend_alignment,
            metrics.z_score,
        )

        # Get sector-algorithm fitness from optimization results
        use_fitness = False
        if self._optimization_results and algorithm:
            fitness = self._optimization_results.get_fitness_score(sector, algorithm)
            metrics.sector_algorithm_fitness = fitness
            use_fitness = True
        elif self._optimization_results:
            # Try to get fitness using best algorithm for sector
            best_algo, _ = self._optimization_results.get_best_algorithm(sector)
            if best_algo:
                fitness = self._optimization_results.get_fitness_score(sector, best_algo)
                metrics.sector_algorithm_fitness = fitness
                use_fitness = True

        # Calculate final confidence
        metrics.calculate_confidence(use_fitness=use_fitness)

        return metrics

    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate normalized momentum strength."""
        if len(prices) < 20:
            return 0.0

        # 20-day return
        ret_20 = (prices[-1] / prices[-20]) - 1

        # Normalize to [-1, 1] range (assuming +-20% is extreme)
        normalized = np.clip(ret_20 / 0.20, -1, 1)

        return float(normalized)

    def _calculate_trend_alignment(self, prices: np.ndarray) -> float:
        """Calculate trend alignment score based on MA positions."""
        if len(prices) < 20:
            return 0.5

        ma_5 = np.mean(prices[-5:])
        ma_10 = np.mean(prices[-10:])
        ma_20 = np.mean(prices[-20:])

        # Score based on MA alignment
        # Perfect uptrend: ma_5 > ma_10 > ma_20
        # Perfect downtrend: ma_5 < ma_10 < ma_20

        score = 0.5  # Neutral

        if ma_5 > ma_10 > ma_20:
            # Uptrend
            strength = (ma_5 - ma_20) / ma_20
            score = 0.5 + min(strength * 5, 0.5)
        elif ma_5 < ma_10 < ma_20:
            # Downtrend
            strength = (ma_20 - ma_5) / ma_20
            score = 0.5 - min(strength * 5, 0.5)

        return float(np.clip(score, 0, 1))

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate 20-day annualized volatility."""
        if len(prices) < 20:
            return 0.2  # Default 20%

        returns = np.diff(prices[-21:]) / prices[-21:-1]
        vol = np.std(returns) * np.sqrt(252)

        return float(vol)

    def _calculate_vol_percentile(self, prices: np.ndarray) -> float:
        """Calculate current vol vs historical percentile."""
        if len(prices) < 60:
            return 0.5

        # Calculate rolling 20-day volatility
        vols = []
        for i in range(40, len(prices)):
            returns = np.diff(prices[i-20:i+1]) / prices[i-20:i]
            vols.append(np.std(returns))

        if not vols:
            return 0.5

        current_vol = vols[-1]
        percentile = np.sum(np.array(vols) <= current_vol) / len(vols)

        return float(percentile)

    def _calculate_strategy_agreement(
        self,
        momentum: float,
        trend: float,
        z_score: float,
    ) -> float:
        """Calculate agreement between different strategy signals."""
        # Momentum suggests direction
        mom_direction = 1 if momentum > 0.05 else (-1 if momentum < -0.05 else 0)

        # Trend suggests direction
        trend_direction = 1 if trend > 0.6 else (-1 if trend < 0.4 else 0)

        # Z-score suggests mean-reversion direction (opposite)
        mr_direction = -1 if z_score > 1.0 else (1 if z_score < -1.0 else 0)

        # Count agreements
        directions = [mom_direction, trend_direction]
        if abs(z_score) > 1.0:
            directions.append(mr_direction)

        non_zero = [d for d in directions if d != 0]
        if not non_zero:
            return 0.5  # No clear signals

        # Check if all non-zero signals agree
        if all(d == non_zero[0] for d in non_zero):
            return 1.0
        elif any(d != non_zero[0] for d in non_zero):
            return 0.3  # Disagreement

        return 0.5


def calculate_position_size(
    confidence: float,
    base_allocation: float,
    min_allocation: float = 0.02,
    max_allocation: float = 0.15,
) -> float:
    """
    Calculate position size based on confidence.

    Uses confidence to scale between min and max allocation.

    Args:
        confidence: Confidence score [0, 1]
        base_allocation: Equal-weight base allocation
        min_allocation: Minimum position size
        max_allocation: Maximum position size

    Returns:
        Position size as fraction of portfolio
    """
    # Scale confidence to position size
    # confidence < 0.3 -> min_allocation
    # confidence > 0.7 -> max_allocation
    # Linear interpolation between

    if confidence < 0.3:
        return min_allocation
    elif confidence > 0.7:
        return max_allocation
    else:
        # Linear interpolation
        scale = (confidence - 0.3) / 0.4
        return min_allocation + scale * (max_allocation - min_allocation)
