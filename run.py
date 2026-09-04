#!/usr/bin/env python
"""
====================================================
QuantAI Professional v5.0
Main CLI Entry Point
====================================================

Usage:
    quantai data download --symbol BTC/USDT --timeframe 15m
    quantai indicators build --input data/raw.csv --output data/prepared.csv
    quantai ml train --dataset data/dataset.parquet --walk-forward
    quantai backtest --prepared data/prepared.csv --ml-enabled
    quantai paper --config config/paper.yaml
    quantai live --config config/live.yaml --dry-run
====================================================
"""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import Settings, settings

app = typer.Typer(
    name="quantai",
    help="QuantAI Professional - AI-driven Cryptocurrency Trading Platform",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


# ============================================================
# GLOBAL OPTIONS
# ============================================================

def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]QuantAI[/bold cyan] version [green]{settings.version}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (YAML/JSON)",
        exists=True,
    ),
    env_file: Optional[Path] = typer.Option(
        None,
        "--env-file",
        "-e",
        help="Path to .env file",
        exists=True,
    ),
):
    """
    QuantAI Professional - Algorithmic Trading Platform

    [bold]Commands:[/bold]
      data        Market data management
      indicators  Technical indicator calculation
      ml          Machine learning pipeline
      backtest    Historical backtesting
      paper       Paper trading simulation
      live        Live trading (requires exchange API)
      config      Configuration management
    """
    if config_file:
        console.print(f"[yellow]Config file:[/yellow] {config_file}")
    if env_file:
        console.print(f"[yellow]Env file:[/yellow] {env_file}")


# ============================================================
# DATA COMMANDS
# ============================================================

data_app = typer.Typer(help="Market data management")
app.add_typer(data_app, name="data")


@data_app.command("download")
def data_download(
    symbol: str = typer.Option("BTC/USDT", "--symbol", "-s", help="Trading symbol"),
    timeframe: str = typer.Option("15m", "--timeframe", "-t", help="Candle timeframe"),
    limit: int = typer.Option(1000, "--limit", "-l", help="Number of candles"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV path"),
    exchange: str = typer.Option("binance", "--exchange", "-x", help="Exchange name"),
):
    """Download historical OHLCV data from exchange"""
    from src.data_loader import load_binance_data

    console.print(f"[cyan]Downloading {symbol} {timeframe} from {exchange}...[/cyan]")

    df = load_binance_data(symbol=symbol, timeframe=timeframe, limit=limit)

    console.print(f"[green]Downloaded {len(df)} candles[/green]")

    if output:
        df.to_csv(output, index=False)
        console.print(f"[green]Saved to {output}[/green]")
    else:
        console.print(df.head(10).to_string())


@data_app.command("validate")
def data_validate(
    input_file: Path = typer.Argument(..., help="Path to CSV/Parquet file"),
):
    """Validate market data file structure"""
    import pandas as pd

    console.print(f"[cyan]Validating {input_file}...[/cyan]")

    if input_file.suffix == ".parquet":
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)

    if missing:
        console.print(f"[red]Missing columns: {missing}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Rows: {len(df)}[/green]")
    console.print(f"[green]Columns: {list(df.columns)}[/green]")
    console.print(f"[green]Date range: {df['timestamp'].min()} to {df['timestamp'].max()}[/green]")
    console.print(f"[green]NaN count: {df.isna().sum().sum()}[/green]")


# ============================================================
# INDICATORS COMMANDS
# ============================================================

indicators_app = typer.Typer(help="Technical indicator calculation")
app.add_typer(indicators_app, name="indicators")


@indicators_app.command("build")
def indicators_build(
    input_file: Path = typer.Argument(..., help="Input OHLCV CSV/Parquet"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file with indicators"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Indicator config YAML"),
):
    """Calculate all technical indicators on OHLCV data"""
    import pandas as pd
    from src.indicators import add_indicators

    console.print(f"[cyan]Loading {input_file}...[/cyan]")

    if input_file.suffix == ".parquet":
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)

    console.print(f"[cyan]Calculating indicators...[/cyan]")
    df = add_indicators(df)

    console.print(f"[green]Added {len(df.columns) - 6} indicator columns[/green]")

    if output.suffix == ".parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False)

    console.print(f"[green]Saved to {output}[/green]")


@indicators_app.command("list")
def indicators_list():
    """List all available indicators"""
    from src.indicators import __all__ as indicator_functions

    table = Table(title="Available Indicators")
    table.add_column("Function", style="cyan")
    table.add_column("Description", style="white")

    descriptions = {
        "sma": "Simple Moving Average",
        "ema": "Exponential Moving Average",
        "rsi": "Relative Strength Index",
        "macd": "Moving Average Convergence Divergence",
        "true_range": "True Range",
        "atr": "Average True Range",
        "adx": "Average Directional Index",
        "bollinger": "Bollinger Bands",
        "vwap": "Volume Weighted Average Price",
        "obv": "On Balance Volume",
        "volume_sma": "Volume Simple Moving Average",
        "supertrend": "SuperTrend",
        "trend_score": "Composite Trend Strength Score",
        "volume_filter": "Abnormal Volume Detector",
        "volatility_filter": "High Volatility Detector",
        "breakout_filter": "Price Breakout Detector",
        "add_indicators": "Calculate all indicators at once",
    }

    for func in sorted(indicator_functions):
        table.add_row(func, descriptions.get(func, ""))

    console.print(table)


# ============================================================
# ML COMMANDS
# ============================================================

ml_app = typer.Typer(help="Machine learning pipeline")
app.add_typer(ml_app, name="ml")


@ml_app.command("train")
def ml_train(
    dataset: Path = typer.Argument(..., help="Path to training dataset (CSV/Parquet)"),
    model_output: Optional[Path] = typer.Option(None, "--output", "-o", help="Model output path"),
    walk_forward: bool = typer.Option(False, "--walk-forward", "-w", help="Use walk-forward validation"),
    n_splits: int = typer.Option(5, "--n-splits", help="Number of CV folds"),
    embargo_pct: float = typer.Option(0.01, "--embargo", help="Embargo percentage for PurgedKFold"),
):
    """Train XGBoost model on prepared dataset"""
    import pandas as pd
    from src.ml_engine import MLEngine, MLConfig, train_model

    console.print(f"[cyan]Loading dataset from {dataset}...[/cyan]")

    if dataset.suffix == ".parquet":
        df = pd.read_parquet(dataset)
    else:
        df = pd.read_csv(dataset)

    config = MLConfig(
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        cv_type="purged",
    )

    console.print(f"[cyan]Training with PurgedKFold (n_splits={n_splits}, embargo={embargo_pct})...[/cyan]")

    engine, result = train_model(df)

    if model_output:
        engine.model_manager.save(engine.model, model_output)
        console.print(f"[green]Model saved to {model_output}[/green]")

    console.print(f"[green]Training complete![/green]")
    console.print(f"  Accuracy: {result.accuracy:.4f}")
    console.print(f"  Balanced Accuracy: {result.balanced_accuracy:.4f}")
    console.print(f"  F1 Score: {result.f1:.4f}")


@ml_app.command("walk-forward")
def ml_walk_forward(
    prepared: Path = typer.Argument(..., help="Path to prepared data with indicators"),
    train_size: int = typer.Option(500, "--train-size", help="Training window size"),
    test_size: int = typer.Option(100, "--test-size", help="Test window size"),
    step_size: Optional[int] = typer.Option(None, "--step-size", help="Step size (default=test_size)"),
    initial_balance: float = typer.Option(1000.0, "--balance", "-b", help="Initial balance"),
    n_splits: int = typer.Option(5, "--n-splits", help="PurgedKFold n_splits"),
    embargo_pct: float = typer.Option(0.01, "--embargo", help="Embargo percentage"),
    purge_pct: float = typer.Option(0.0, "--purge", help="Purge percentage"),
    cv_type: str = typer.Option("purged", "--cv-type", help="CV type: purged or combinatorial"),
    future_bars: int = typer.Option(5, "--future-bars", help="Future bars for target"),
    target_profit: float = typer.Option(0.002, "--target-profit", help="Target profit for labeling"),
    warmup_bars: int = typer.Option(200, "--warmup", help="Warmup bars for indicators"),
    save_models: bool = typer.Option(True, "--save-models/--no-save-models", help="Save model per window"),
):
    """Run ML Walk-Forward with PurgedKFold CV in each training window"""
    import pandas as pd
    from src.ml_walk_forward import MLWalkForwardEngine, MLConfig, DatasetConfig, run_ml_walk_forward
    
    console.print(f"[cyan]Loading prepared data from {prepared}...[/cyan]")
    
    if prepared.suffix == ".parquet":
        df = pd.read_parquet(prepared)
    else:
        df = pd.read_csv(prepared)
    
    ml_config = MLConfig(
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        purge_pct=purge_pct,
        cv_type=cv_type,
    )
    
    dataset_config = DatasetConfig(
        future_bars=future_bars,
        target_profit=target_profit,
        warmup_bars=warmup_bars,
    )
    
    console.print(f"[cyan]Running ML Walk-Forward (train={train_size}, test={test_size}, step={step_size or test_size})...[/cyan]")
    console.print(f"[cyan]PurgedKFold: {cv_type}, n_splits={n_splits}, embargo={embargo_pct}, purge={purge_pct}[/cyan]")
    
    result = run_ml_walk_forward(
        df,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        initial_balance=initial_balance,
        ml_config=ml_config,
        dataset_config=dataset_config,
    )
    
    console.print(f"[green]ML Walk-Forward complete![/green]")
    console.print(f"  Windows: {result.total_windows}")
    console.print(f"  Models Trained: {result.models_trained}")
    console.print(f"  Avg Balanced Accuracy: {result.avg_balanced_accuracy:.4f}")
    console.print(f"  Avg F1: {result.avg_f1:.4f}")
    console.print(f"  Net Profit: {result.net_profit:.2f}")
    console.print(f"  Win Rate: {result.win_rate:.2f}%")


@ml_app.command("predict")
def ml_predict(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    features: Path = typer.Argument(..., help="Path to features CSV/Parquet"),
):
    """Generate predictions using trained model"""
    import pandas as pd
    from src.model_manager import ModelManager
    from src.feature_engine import build_features

    console.print(f"[cyan]Loading model from {model_path}...[/cyan]")

    manager = ModelManager()
    manager.model_path = model_path
    model = manager.load()

    if model is None:
        console.print("[red]Failed to load model[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Loading features from {features}...[/cyan]")

    if features.suffix == ".parquet":
        df = pd.read_parquet(features)
    else:
        df = pd.read_csv(features)

    feature_dict = build_features(df)

    import numpy as np
    X = pd.DataFrame([feature_dict])

    probs = model.predict_proba(X)[0]
    pred = np.argmax(probs)

    labels = {0: "SELL", 1: "HOLD", 2: "BUY"}

    console.print(f"[green]Prediction: {labels[pred]}[/green]")
    console.print(f"  SELL: {probs[0]*100:.2f}%")
    console.print(f"  HOLD: {probs[1]*100:.2f}%")
    console.print(f"  BUY:  {probs[2]*100:.2f}%")


@ml_app.command("validate")
def ml_validate(
    dataset: Path = typer.Argument(..., help="Path to dataset"),
    n_splits: int = typer.Option(5, "--n-splits"),
    embargo_pct: float = typer.Option(0.01, "--embargo"),
):
    """Run PurgedKFold cross-validation on dataset"""
    import pandas as pd
    from src.ml_engine import MLEngine, MLConfig
    from src.validation.purged_kfold import get_purged_cv
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    console.print(f"[cyan]Loading dataset...[/cyan]")
    df = pd.read_parquet(dataset) if dataset.suffix == ".parquet" else pd.read_csv(dataset)

    # Prepare features and target (similar to prepare_dataset)
    from src.feature_engine import build_features

    # This is simplified - in reality you'd use the full prepare_dataset
    console.print("[yellow]Simplified validation - use walk-forward for production[/yellow]")

    config = MLConfig(n_splits=n_splits, embargo_pct=embargo_pct)
    engine = MLEngine(config=config)

    # Quick validation split
    cv = get_purged_cv(cv_type="purged", n_splits=n_splits, embargo_pct=embargo_pct)

    console.print(f"[green]PurgedKFold configured: {n_splits} splits, {embargo_pct*100:.1f}% embargo[/green]")


# ============================================================
# BACKTEST COMMANDS
# ============================================================

backtest_app = typer.Typer(help="Historical backtesting")
app.add_typer(backtest_app, name="backtest")


@backtest_app.command("run")
def backtest_run(
    prepared: Path = typer.Argument(..., help="Path to prepared data with indicators"),
    ml_enabled: bool = typer.Option(False, "--ml-enabled", help="Enable ML predictions"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Results output path"),
    initial_balance: float = typer.Option(1000.0, "--balance", "-b", help="Initial balance"),
):
    """Run historical backtest"""
    import pandas as pd
    from src.backtest_engine import BacktestEngine

    console.print(f"[cyan]Loading prepared data from {prepared}...[/cyan]")

    if prepared.suffix == ".parquet":
        df = pd.read_parquet(prepared)
    else:
        df = pd.read_csv(prepared)

    engine = BacktestEngine(initial_balance=initial_balance)
    result = engine.run(df)

    BacktestEngine.print_report(result)

    if output:
        import json
        output_data = {
            "initial_balance": result.initial_balance,
            "final_balance": result.final_balance,
            "net_profit": result.net_profit,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "trades": result.trades,
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        console.print(f"[green]Results saved to {output}[/green]")


@backtest_app.command("walk-forward")
def backtest_walk_forward(
    prepared: Path = typer.Argument(..., help="Path to prepared data"),
    train_size: int = typer.Option(500, "--train-size"),
    test_size: int = typer.Option(100, "--test-size"),
    step_size: Optional[int] = typer.Option(None, "--step-size"),
    initial_balance: float = typer.Option(1000.0, "--balance"),
):
    """Run walk-forward validation"""
    import pandas as pd
    from src.walk_forward_engine import WalkForwardEngine

    console.print(f"[cyan]Loading prepared data...[/cyan]")
    df = pd.read_parquet(prepared) if prepared.suffix == ".parquet" else pd.read_csv(prepared)

    engine = WalkForwardEngine(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size or test_size,
        initial_balance=initial_balance,
    )

    console.print(f"[cyan]Running walk-forward (train={train_size}, test={test_size})...[/cyan]")
    result = engine.run(df)

    console.print(f"[green]Walk-forward complete: {len(result.windows)} windows[/green]")
    console.print(f"  Net Profit: {result.net_profit:.2f}")
    console.print(f"  Total Trades: {result.total_trades}")
    console.print(f"  Win Rate: {result.win_rate:.2f}%")


# ============================================================
# PAPER TRADING COMMANDS
# ============================================================

paper_app = typer.Typer(help="Paper trading simulation")
app.add_typer(paper_app, name="paper")


@paper_app.command("run")
def paper_run(
    config: Path = typer.Option(..., "--config", "-c", help="Paper trading config YAML"),
    duration: Optional[int] = typer.Option(None, "--duration", "-d", help="Duration in minutes"),
):
    """Run paper trading session"""
    console.print(f"[cyan]Starting paper trading with config {config}...[/cyan]")
    
    import asyncio
    from src.lifecycle import startup, shutdown
    
    async def run_paper_trading():
        state = await startup()
        # Override mode to PAPER
        state.settings.config.mode = "PAPER"
        
        # Import and start paper trading engine
        from src.paper_trading_runner import run_paper_trading
        
        console.print(f"[green]Paper trading started[/green]")
        
        try:
            # Run paper trading
            await run_paper_trading(
                state=state,
                duration_minutes=duration,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        finally:
            await shutdown(state)
    
    asyncio.run(run_paper_trading())


# ============================================================
# LIVE COMMANDS
# ============================================================

live_app = typer.Typer(help="Live trading (requires exchange API)")
app.add_typer(live_app, name="live")


@live_app.command("start")
def live_start(
    config: Path = typer.Option(..., "--config", "-c", help="Live trading config YAML"),
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Dry run mode (default)"),
):
    """Start live trading"""
    mode = "DRY RUN" if dry_run else "LIVE"
    console.print(f"[cyan]Starting {mode} trading with config {config}...[/cyan]")
    console.print("[red]Live trading not yet implemented[/red]")


# ============================================================
# CONFIG COMMANDS
# ============================================================

config_app = typer.Typer(help="Configuration management")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    section: Optional[str] = typer.Option(None, "--section", "-s", help="Show specific section"),
):
    """Show current configuration"""
    table = Table(title="QuantAI Configuration")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="yellow")
    table.add_column("Value", style="green")

    def add_section(prefix: str, obj):
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                table.add_row(prefix, key, str(value))

    if section is None or section == "exchange":
        add_section("exchange", settings.exchange)
    if section is None or section == "account":
        add_section("account", settings.account)
    if section is None or section == "indicators":
        add_section("indicators", settings.indicators)
    if section is None or section == "ml":
        add_section("ml", settings.ml)
    if section is None or section == "risk":
        add_section("risk", settings.risk)

    console.print(table)


@config_app.command("validate")
def config_validate():
    """Validate current configuration"""
    console.print("[cyan]Validating configuration...[/cyan]")

    errors = []

    if settings.account.initial_balance <= 0:
        errors.append("initial_balance must be > 0")

    if not 0 < settings.risk.risk_per_trade <= 1:
        errors.append("risk_per_trade must be in (0, 1]")

    if settings.exchange.limit < 100:
        errors.append("limit must be >= 100")

    if settings.ml.ml_enabled and not Path(settings.ml.model_path).exists():
        errors.append(f"Model file not found: {settings.ml.model_path}")

    if errors:
        console.print("[red]Validation failed:[/red]")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(1)

    console.print("[green]Configuration valid![/green]")


@config_app.command("export-env")
def config_export_env(
    output: Path = typer.Option(Path(".env"), "--output", "-o", help="Output .env file"),
):
    """Export current settings to .env file"""
    import os

    env_lines = [
        f"# QuantAI Configuration Export",
        f"# Generated at {pd.Timestamp.now()}",
        "",
    ]

    def export_section(prefix: str, obj):
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                env_key = f"{prefix}__{key.upper()}"
                env_lines.append(f"{env_key}={value}")

    export_section("EXCHANGE", settings.exchange)
    export_section("ACCOUNT", settings.account)
    export_section("COMMISSION", settings.commission)
    export_section("INDICATORS", settings.indicators)
    export_section("BACKTEST", settings.backtest)
    export_section("ML", settings.ml)
    export_section("TELEGRAM", settings.telegram)
    export_section("LOGGING", settings.logging)
    export_section("RISK", settings.risk)

    output.write_text("\n".join(env_lines))
    console.print(f"[green]Exported to {output}[/green]")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app()