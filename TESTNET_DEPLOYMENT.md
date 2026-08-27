# QuantAI Testnet Deployment Guide

## Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- Binance Testnet API credentials
- 4GB+ RAM, 2+ CPU cores
- 20GB+ disk space

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/DepthSight-Pro/QuantAI.git
cd QuantAI

# 2. Configure environment
cp .env.testnet.template .env.testnet
# Edit .env.testnet with your Binance Testnet API keys

# 3. Deploy
./deploy_testnet.sh up
```

## Configuration

### Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `BINANCE_TESTNET_API_KEY` | Binance Testnet API Key | Yes |
| `BINANCE_TESTNET_API_SECRET` | Binance Testnet API Secret | Yes |
| `BINANCE_TESTNET` | Set to `true` for testnet | Yes |

### Optional Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SYMBOL` | `BTCUSDT` | Trading pair |
| `TIMEFRAME` | `15m` | Candle timeframe |
| `INITIAL_BALANCE` | `10000.0` | Starting paper balance |
| `RISK_PER_TRADE` | `0.005` | Risk per trade (0.5%) |
| `MAX_DRAWDOWN_PCT` | `5.0` | Max drawdown % |
| `ML_ENABLED` | `true` | Enable ML predictions |

## Services

| Service | Port | Description |
|---------|------|-------------|
| QuantAI | 9090 | Main application + metrics |
| Prometheus | 9091 | Metrics collection |
| Grafana | 3000 | Dashboards (admin/admin) |
| Alertmanager | 9093 | Alert management |
| Redis | 6379 | Caching |
| PostgreSQL | 5432 | Persistent storage |

## Access URLs

- **QuantAI Metrics**: http://localhost:9090/metrics
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091
- **Alertmanager**: http://localhost:9093

## Common Commands

```bash
# Start services
./deploy_testnet.sh up

# View logs
./deploy_testnet.sh logs quantai-testnet

# Check status
./deploy_testnet.sh status

# View logs for specific service
./deploy_testnet.sh logs prometheus

# Restart services
./deploy_testnet.sh restart

# Stop all services
./deploy_testnet.sh down

# Backup data
./deploy_testnet.sh backup

# Restore from backup
./deploy_testnet.sh restore ./backups/20240115_120000

# Update to latest version
./deploy_testnet.sh update

# Clean up everything
./deploy_testnet.sh cleanup
```

## Health Checks

```bash
# Check all services
./deploy_testnet.sh health

# Check specific service
curl http://localhost:9090/health/live
curl http://localhost:9090/health/ready
```

## Monitoring

### Grafana Dashboards
1. Open http://localhost:3000
2. Login: admin / admin (or GRAFANA_PASSWORD)
3. Import dashboards from `monitoring/grafana/dashboards/`

### Key Metrics to Monitor

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `quantai_balance_usdt` | < 5000 | Low balance |
| `quantai_win_rate` | < 40% | Low win rate |
| `quantai_risk_drawdown_pct` | > 10% | High drawdown |
| `quantai_open_positions` | > 5 | Too many positions |
| `quantai_ml_balanced_accuracy` | < 50% | ML model degraded |

## Backup & Restore

```bash
# Create backup
./deploy_testnet.sh backup

# List backups
ls -la backups/

# Restore from backup
./deploy_testnet.sh restore ./backups/20240115_120000
```

## Troubleshooting

### Service won't start
```bash
# Check logs
./deploy_testnet.sh logs quantai-testnet

# Check config
./deploy_testnet.sh config validate
```

### High memory usage
```bash
# Check memory usage
docker stats

# Restart with limits
docker-compose -f docker-compose.testnet.yml restart quantai-testnet
```

### API connection issues
```bash
# Test Binance connectivity
curl -X GET "https://testnet.binancefuture.com/fapi/v1/ping"

# Check API key permissions
curl -H "X-MBX-APIKEY: $BINANCE_TESTNET_API_KEY" \
  "https://testnet.binancefuture.com/fapi/v2/account"
```

## Security Checklist

- [ ] API keys stored in `.env.testnet` (not in code)
- [ ] Testnet mode enabled (`BINANCE_TESTNET=true`)
- [ ] No real funds at risk (testnet only)
- [ ] Firewall: only necessary ports open
- [ ] SSH keys for server access
- [ ] Regular backups scheduled
- [ ] Monitoring alerts configured

## Updating

```bash
# Pull latest code
git pull origin main

# Update containers
./deploy_testnet.sh update

# Or manually:
./deploy_testnet.sh pull
./deploy_testnet.sh build
./deploy_testnet.sh restart
```

## Support

- Check logs: `./deploy_testnet.sh logs`
- Health check: `./deploy_testnet.sh health`
- GitHub Issues: https://github.com/DepthSight-Pro/QuantAI/issues