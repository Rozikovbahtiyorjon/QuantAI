#!/bin/bash
# =========================================================
# QuantAI Testnet Deployment Script
# =========================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.testnet.yml"
ENV_FILE=".env.testnet"
PROJECT_NAME="quantai-testnet"

# =========================================================
# Helper Functions
# =========================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file $ENV_FILE not found!"
        log_info "Copy .env.testnet.template to .env.testnet and fill in your values"
        exit 1
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose not installed"
        exit 1
    fi
}

check_env_vars() {
    local required_vars=(
        "BINANCE_TESTNET_API_KEY"
        "BINANCE_TESTNET_API_SECRET"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ] || [ "${!var}" == "your_testnet_api_key_here" ] || [ "${!var}" == "your_testnet_api_secret_here" ]; then
            log_error "Required environment variable $var is not set or has default value"
            return 1
        fi
    done
}

# =========================================================
# Main Commands
# =========================================================

cmd_build() {
    log_info "Building QuantAI testnet image..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build --no-cache
    log_success "Build completed"
}

cmd_up() {
    log_info "Starting QuantAI testnet stack..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    log_success "Stack started"
    log_info "Waiting for services to be healthy..."
    sleep 10
    check_health
}

cmd_down() {
    log_info "Stopping QuantAI testnet stack..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
    log_success "Stack stopped"
}

cmd_restart() {
    cmd_down
    cmd_up
}

cmd_logs() {
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f --tail=100
}

cmd_status() {
    log_info "Service Status:"
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
    
    echo ""
    log_info "Health Checks:"
    check_health
}

check_health() {
    local services=("quantai-testnet" "prometheus" "grafana" "redis" "postgres" "redis")
    
    for service in "${services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            log_success "$service: RUNNING"
        else
            log_error "$service: NOT RUNNING"
        fi
    done
}

cmd_health() {
    check_health
}

cmd_logs() {
    local service=${1:-quantai-testnet}
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f --tail=100 "$service"
}

cmd_pull() {
    log_info "Pulling latest images..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
    log_success "Images pulled"
}

cmd_update() {
    log_info "Updating QuantAI testnet..."
    cmd_pull
    cmd_build
    cmd_restart
    log_success "Update completed"
}

cmd_cleanup() {
    log_warning "This will remove all containers, volumes, and networks!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down -v --remove-orphans
        docker system prune -f
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

cmd_backup() {
    local backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    log_info "Creating backup in $backup_dir..."
    
    # Backup PostgreSQL
    if docker ps --filter "name=postgres" --filter "status=running" | grep -q postgres; then
        log_info "Backing up PostgreSQL..."
        docker exec quantai-postgres pg_dump -U quantai quantai > "$backup_dir/postgres_backup.sql"
    fi
    
    # Backup data directory
    if [ -d "./data" ]; then
        cp -r ./data "$backup_dir/data"
    fi
    
    # Backup models
    if [ -d "./models" ]; then
        cp -r ./models "$backup_dir/models"
    fi
    
    # Backup checkpoints
    if [ -d "./checkpoints" ]; then
        cp -r ./checkpoints "$backup_dir/checkpoints"
    fi
    
    # Backup config
    cp -r ./config "$backup_dir/config"
    
    # Backup environment
    cp "$ENV_FILE" "$backup_dir/.env.testnet"
    
    log_success "Backup created at $backup_dir"
}

cmd_restore() {
    local backup_dir=${1:-}
    
    if [ -z "$backup_dir" ]; then
        log_error "Usage: $0 restore <backup_directory>"
        exit 1
    fi
    
    if [ ! -d "$backup_dir" ]; then
        log_error "Backup directory $backup_dir not found"
        exit 1
    fi
    
    log_warning "This will restore data from $backup_dir. Current data will be overwritten!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    log_info "Restoring from $backup_dir..."
    
    # Stop services first
    cmd_down
    
    # Restore PostgreSQL
    if [ -f "$backup_dir/postgres_backup.sql" ]; then
        log_info "Restoring PostgreSQL..."
        cmd_up
        sleep 10
        docker exec -i quantai-postgres psql -U quantai quantai < "$backup_dir/postgres_backup.sql"
    fi
    
    # Restore data
    if [ -d "$backup_dir/data" ]; then
        rm -rf ./data
        cp -r "$backup_dir/data" ./
    fi
    
    # Restore models
    if [ -d "$backup_dir/models" ]; then
        rm -rf ./models
        cp -r "$backup_dir/models" ./
    fi
    
    # Restore checkpoints
    if [ -d "$backup_dir/checkpoints" ]; then
        rm -rf ./checkpoints
        cp -r "$backup_dir/checkpoints" ./
    fi
    
    # Restart
    cmd_up
    log_success "Restore completed"
}

# =========================================================
# Main
# =========================================================

show_help() {
    echo "QuantAI Testnet Deployment Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker images"
    echo "  up          Start all services"
    echo "  down        Stop all services"
    echo "  restart     Restart all services"
    echo "  logs        View logs (service optional)"
    echo "  status      Show service status"
    echo "  health      Check service health"
    echo "  pull        Pull latest images"
    echo "  update      Update and restart (pull + build + restart)"
    echo "  backup      Create backup"
    echo "  restore     Restore from backup"
    echo "  cleanup     Remove all containers, volumes, networks"
    echo ""
    echo "Environment file: $ENV_FILE"
    echo "Compose file: $COMPOSE_FILE"
}

main() {
    case "${1:-help}" in
        build)
            check_docker
            check_env_file
            cmd_build
            ;;
        up)
            check_docker
            check_env_file
            check_env_vars
            cmd_up
            ;;
        down)
            check_docker
            check_env_file
            cmd_down
            ;;
        restart)
            check_docker
            check_env_file
            cmd_restart
            ;;
        logs)
            check_docker
            check_env_file
            cmd_logs "${2:-}"
            ;;
        status)
            check_docker
            check_env_file
            cmd_status
            ;;
        health)
            check_docker
            check_env_file
            cmd_health
            ;;
        pull)
            check_docker
            check_env_file
            cmd_pull
            ;;
        update)
            check_docker
            check_env_file
            cmd_update
            ;;
        backup)
            check_docker
            check_env_file
            cmd_backup
            ;;
        restore)
            check_docker
            check_env_file
            cmd_restore "${2:-}"
            ;;
        cleanup)
            check_docker
            check_env_file
            cmd_cleanup
            ;;
        *)
            show_help
            ;;
    esac
}

# Run main
main "$@"