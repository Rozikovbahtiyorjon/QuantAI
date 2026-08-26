"""
QuantAI Self-Hosted Vault

Secure API key management with PostgreSQL + Fernet encryption.

Features:
- Fernet symmetric encryption for API keys at rest
- PostgreSQL backend for persistent, queryable storage
- Key rotation support
- Access logging and audit trail
- TTL-based key expiration
- Role-based access control (read/write/admin)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import asyncpg
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND TYPES
# ============================================================

class KeyRole(Enum):
    """API key roles/permissions."""
    READ = "read"           # Read-only access (market data, account info)
    WRITE = "write"         # Write access (place orders, modify positions)
    ADMIN = "admin"         # Full access (key management, user management)


class KeyStatus(Enum):
    """API key lifecycle status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_ROTATION = "pending_rotation"


@dataclass
class VaultConfig:
    """Vault configuration."""
    
    # PostgreSQL connection
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "quantai_vault"
    db_user: str = "quantai"
    db_password: str = "changeme"
    db_pool_size: int = 10
    
    # Encryption
    master_key: Optional[str] = None  # Base64-encoded 32-byte key
    key_derivation_iterations: int = 100000
    
    # Key management
    default_ttl_days: int = 90
    max_key_age_days: int = 365
    rotation_warning_days: int = 7
    
    # Security
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    @classmethod
    def from_env(cls) -> "VaultConfig":
        """Create config from environment variables."""
        return cls(
            db_host=os.getenv("VAULT_DB_HOST", "localhost"),
            db_port=int(os.getenv("VAULT_DB_PORT", "5432")),
            db_name=os.getenv("VAULT_DB_NAME", "quantai_vault"),
            db_user=os.getenv("VAULT_DB_USER", "quantai"),
            db_password=os.getenv("VAULT_DB_PASSWORD", "changeme"),
            db_pool_size=int(os.getenv("VAULT_DB_POOL_SIZE", "10")),
            master_key=os.getenv("VAULT_MASTER_KEY"),
            default_ttl_days=int(os.getenv("VAULT_DEFAULT_TTL_DAYS", "90")),
            max_key_age_days=int(os.getenv("VAULT_MAX_KEY_AGE_DAYS", "365")),
            rotation_warning_days=int(os.getenv("VAULT_ROTATION_WARNING_DAYS", "7")),
        )


class KeyRoleDB(str, Enum):
    """Database representation of key roles."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class APIKey:
    """API key record."""
    
    id: str
    name: str
    exchange: str
    encrypted_key: str
    encrypted_secret: str
    encrypted_passphrase: Optional[str] = None
    
    role: KeyRole = KeyRole.READ
    status: KeyStatus = KeyStatus.ACTIVE
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Access control
    allowed_ips: List[str] = field(default_factory=list)  # CIDR notation
    allowed_exchanges: List[str] = field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    rotated_at: Optional[datetime] = None
    
    # Rotation
    rotation_count: int = 0
    previous_key_id: Optional[str] = None
    
    # Usage tracking
    usage_count: int = 0
    last_error: Optional[str] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_locked(self) -> bool:
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def can_access(self, exchange: str, ip: str, required_role: KeyRole) -> bool:
        """Check if key can be used for given context."""
        if self.status != KeyStatus.ACTIVE:
            return False
        if self.is_expired():
            return False
        if self.is_locked():
            return False
        
        # Role check
        role_hierarchy = {KeyRole.READ: 1, KeyRole.WRITE: 2, KeyRole.ADMIN: 3}
        if role_hierarchy[self.role] < role_hierarchy[required_role]:
            return False
        
        # Exchange check
        if self.allowed_exchanges and exchange not in self.allowed_exchanges:
            return False
        
        # IP check (simplified - would use ipaddress module in production)
        if self.allowed_ips:
            # Simplified check - in production use ipaddress.ip_network
            pass
        
        return True


@dataclass
class KeyAccessLog:
    """Audit log for key access."""
    
    id: str
    key_id: str
    action: str  # "read", "write", "rotate", "revoke", "failed"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# ENCRYPTION SERVICE
# ============================================================

class EncryptionService:
    """Handles encryption/decryption using Fernet."""
    
    def __init__(self, master_key: bytes):
        self._fernet = Fernet(master_key)
    
    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> "EncryptionService":
        """Create encryption service from password using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return cls(key)
    
    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a new Fernet key."""
        return Fernet.generate_key()
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt string and return base64-encoded ciphertext."""
        ciphertext = self._fernet.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(ciphertext).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt base64-encoded ciphertext and return plaintext."""
        data = base64.urlsafe_b64decode(ciphertext.encode())
        return self._fernet.decrypt(data).decode()
    
    def encrypt_bytes(self, data: bytes) -> str:
        """Encrypt bytes and return base64-encoded ciphertext."""
        ciphertext = self._fernet.encrypt(data)
        return base64.urlsafe_b64encode(ciphertext).decode()
    
    def decrypt_bytes(self, ciphertext: str) -> bytes:
        """Decrypt base64-encoded ciphertext and return bytes."""
        data = base64.urlsafe_b64decode(ciphertext.encode())
        return self._fernet.decrypt(data)


# ============================================================
# VAULT SERVICE
# ============================================================

class VaultService:
    """
    Self-hosted vault for API key management.
    
    Provides secure storage, retrieval, rotation, and audit
    of API keys with PostgreSQL backend and Fernet encryption.
    """
    
    def __init__(self, config: VaultConfig):
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._encryption: Optional[EncryptionService] = None
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize encryption
        if config.master_key:
            master_key = base64.urlsafe_b64decode(config.master_key)
        else:
            # Generate from environment or create new
            master_key = Fernet.generate_key()
            logger.warning("No master key provided, generated ephemeral key")
        self._encryption = EncryptionService(master_key)
    
    async def initialize(self) -> None:
        """Initialize database connection pool and schema."""
        dsn = (
            f"postgresql://{self.config.db_user}:{self.config.db_password}@"
            f"{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        )
        
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=self.config.db_pool_size,
            command_timeout=60,
        )
        
        await self._create_schema()
        logger.info("Vault service initialized")
    
    async def _create_schema(self) -> None:
        """Create database tables if they don't exist."""
        async with self._pool.acquire() as conn:
            # API keys table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id VARCHAR(64) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    exchange VARCHAR(64) NOT NULL,
                    encrypted_key TEXT NOT NULL,
                    encrypted_secret TEXT NOT NULL,
                    encrypted_passphrase TEXT,
                    role VARCHAR(16) NOT NULL DEFAULT 'read',
                    status VARCHAR(16) NOT NULL DEFAULT 'active',
                    description TEXT DEFAULT '',
                    tags TEXT[] DEFAULT '{}',
                    allowed_ips TEXT[] DEFAULT '{}',
                    allowed_exchanges TEXT[] DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ,
                    last_used_at TIMESTAMPTZ,
                    rotated_at TIMESTAMPTZ,
                    rotation_count INTEGER DEFAULT 0,
                    previous_key_id VARCHAR(64),
                    usage_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMPTZ
                )
            """)
            
            # Access log table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS key_access_log (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    key_id VARCHAR(64) NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
                    action VARCHAR(32) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ip INET,
                    user_agent TEXT,
                    success BOOLEAN NOT NULL DEFAULT TRUE,
                    error_message TEXT,
                    metadata JSONB DEFAULT '{}'
                )
            """)
            
            # Indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_exchange ON api_keys(exchange);
                CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status);
                CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON api_keys(expires_at);
                CREATE INDEX IF NOT EXISTS idx_access_log_key_id ON key_access_log(key_id);
                CREATE INDEX IF NOT EXISTS idx_access_log_timestamp ON key_access_log(timestamp);
            """)
    
    async def start(self) -> None:
        """Start background tasks."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Vault service started")
    
    async def stop(self) -> None:
        """Stop background tasks and close pool."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        if self._pool:
            await self._pool.close()
        logger.info("Vault service stopped")
    
    async def _cleanup_loop(self):
        """Background cleanup of expired keys and logs."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_expired_keys()
                await self._cleanup_old_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Vault cleanup error: {e}")
    
    async def _cleanup_expired_keys(self):
        """Mark expired keys as expired status."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE api_keys 
                SET status = 'expired', updated_at = NOW()
                WHERE status = 'active' 
                AND expires_at IS NOT NULL 
                AND expires_at < NOW()
            """)
    
    async def _cleanup_old_logs(self):
        """Remove access logs older than 90 days."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM key_access_log 
                WHERE timestamp < NOW() - INTERVAL '90 days'
            """)
    
    # ============================================================
    # KEY MANAGEMENT
    # ============================================================
    
    async def create_key(
        self,
        *,
        name: str,
        exchange: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        role: KeyRole = KeyRole.READ,
        description: str = "",
        ttl_days: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        allowed_exchanges: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> APIKey:
        """Create a new API key."""
        
        key_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        ttl = ttl_days or self.config.default_ttl_days
        expires_at = datetime.now(timezone.utc) + timedelta(days=ttl)
        
        # Encrypt sensitive data
        encrypted_key = self._encryption.encrypt(api_key)
        encrypted_secret = self._encryption.encrypt(api_secret)
        encrypted_passphrase = self._encryption.encrypt(passphrase) if passphrase else None
        
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO api_keys (
                    id, name, exchange, encrypted_key, encrypted_secret, encrypted_passphrase,
                    role, status, description, tags, allowed_ips, allowed_exchanges,
                    created_at, updated_at, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """, key_id, name, exchange, encrypted_key, encrypted_secret, encrypted_passphrase,
                role.value, KeyStatus.ACTIVE.value, description,
                tags or [], allowed_ips or [], allowed_exchanges or [],
                datetime.now(timezone.utc), datetime.now(timezone.utc), expires_at)
        
        key = APIKey(
            id=key_id,
            name=name,
            exchange=exchange,
            encrypted_key=encrypted_key,
            encrypted_secret=encrypted_secret,
            encrypted_passphrase=encrypted_passphrase,
            role=role,
            status=KeyStatus.ACTIVE,
            description=description,
            tags=tags or [],
            allowed_ips=allowed_ips or [],
            allowed_exchanges=allowed_exchanges or [],
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )
        
        await self._log_access(key_id, "create", success=True)
        return key
    
    async def get_key(self, key_id: str, decrypt: bool = False) -> Optional[APIKey]:
        """Get API key by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM api_keys WHERE id = $1
            """, key_id)
        
        if not row:
            return None
        
        key = self._row_to_key(row)
        
        if decrypt:
            key.api_key = self._encryption.decrypt(row["encrypted_key"])
            key.api_secret = self._encryption.decrypt(row["encrypted_secret"])
            if row["encrypted_passphrase"]:
                key.api_passphrase = self._encryption.decrypt(row["encrypted_passphrase"])
        
        return key
    
    def _row_to_key(self, row: asyncpg.Record) -> APIKey:
        """Convert database row to APIKey object."""
        return APIKey(
            id=row["id"],
            name=row["name"],
            exchange=row["exchange"],
            encrypted_key=row["encrypted_key"],
            encrypted_secret=row["encrypted_secret"],
            encrypted_passphrase=row["encrypted_passphrase"],
            role=KeyRole(row["role"]),
            status=KeyStatus(row["status"]),
            description=row["description"] or "",
            tags=list(row["tags"]) if row["tags"] else [],
            allowed_ips=list(row["allowed_ips"]) if row["allowed_ips"] else [],
            allowed_exchanges=list(row["allowed_exchanges"]) if row["allowed_exchanges"] else [],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            last_used_at=row["last_used_at"],
            rotated_at=row["rotated_at"],
            rotation_count=row["rotation_count"],
            previous_key_id=row["previous_key_id"],
            usage_count=row["usage_count"],
            last_error=row["last_error"],
            failed_attempts=row["failed_attempts"],
            locked_until=row["locked_until"],
        )
    
    async def list_keys(
        self,
        exchange: Optional[str] = None,
        status: Optional[KeyStatus] = None,
        role: Optional[KeyRole] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[APIKey]:
        """List API keys with filters."""
        conditions = []
        params = []
        param_idx = 1
        
        if exchange:
            conditions.append(f"exchange = ${param_idx}")
            params.append(exchange)
            param_idx += 1
        
        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status.value)
            param_idx += 1
        
        if role:
            conditions.append(f"role = ${param_idx}")
            params.append(role.value)
            param_idx += 1
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT * FROM api_keys
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [self._row_to_key(row) for row in rows]
    
    async def update_key(
        self,
        key_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        role: Optional[KeyRole] = None,
        status: Optional[KeyStatus] = None,
        allowed_ips: Optional[List[str]] = None,
        allowed_exchanges: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        extend_ttl_days: Optional[int] = None,
    ) -> Optional[APIKey]:
        """Update key metadata."""
        
        updates = []
        params = [key_id]
        param_idx = 2
        
        if name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(name)
            param_idx += 1
        
        if description is not None:
            updates.append(f"description = ${param_idx}")
            params.append(description)
            param_idx += 1
        
        if role:
            updates.append(f"role = ${param_idx}")
            params.append(role.value)
            param_idx += 1
        
        if status:
            updates.append(f"status = ${param_idx}")
            params.append(status.value)
            param_idx += 1
        
        if allowed_ips is not None:
            updates.append(f"allowed_ips = ${param_idx}")
            params.append(allowed_ips)
            param_idx += 1
        
        if allowed_exchanges is not None:
            updates.append(f"allowed_exchanges = ${param_idx}")
            params.append(allowed_exchanges)
            param_idx += 1
        
        if tags is not None:
            updates.append(f"tags = ${param_idx}")
            params.append(tags)
            param_idx += 1
        
        if extend_ttl_days:
            updates.append(f"expires_at = expires_at + INTERVAL '{extend_ttl_days} days'")
        
        if not updates:
            return await self.get_key(key_id)
        
        updates.append(f"updated_at = NOW()")
        
        query = f"""
            UPDATE api_keys SET {", ".join(updates)}
            WHERE id = $1
        """
        
        async with self._pool.acquire() as conn:
            await conn.execute(query, *params)
            await self._log_access(key_id, "update", success=True)
        
        return await self.get_key(key_id)
    
    async def rotate_key(
        self,
        key_id: str,
        new_api_key: str,
        new_api_secret: str,
        new_passphrase: Optional[str] = None,
    ) -> Optional[APIKey]:
        """Rotate API key credentials."""
        
        old_key = await self.get_key(key_id, decrypt=True)
        if not old_key:
            return None
        
        new_key_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = old_key.expires_at or (datetime.now(timezone.utc) + timedelta(days=self.config.default_ttl_days))
        
        # Encrypt new credentials
        encrypted_key = self._encryption.encrypt(new_api_key)
        encrypted_secret = self._encryption.encrypt(new_api_secret)
        encrypted_passphrase = self._encryption.encrypt(new_passphrase) if new_passphrase else None
        
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Create new key record
                await conn.execute("""
                    INSERT INTO api_keys (
                        id, name, exchange, encrypted_key, encrypted_secret, encrypted_passphrase,
                        role, status, description, tags, allowed_ips, allowed_exchanges,
                        created_at, updated_at, expires_at, rotated_at,
                        rotation_count, previous_key_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $14, $15, $16, $17)
                """, new_key_id, old_key.name, old_key.exchange,
                    encrypted_key, encrypted_secret, encrypted_passphrase,
                    old_key.role.value, KeyStatus.ACTIVE.value,
                    old_key.description, old_key.tags, old_key.allowed_ips, old_key.allowed_exchanges,
                    datetime.now(timezone.utc), datetime.now(timezone.utc),
                    old_key.expires_at, now, old_key.rotation_count + 1, old_key.id)
                
                # Mark old key as rotated
                await conn.execute("""
                    UPDATE api_keys 
                    SET status = 'revoked', updated_at = NOW(), rotated_at = NOW()
                    WHERE id = $1
                """, key_id)
        
        new_key = APIKey(
            id=new_key_id,
            name=old_key.name,
            exchange=old_key.exchange,
            encrypted_key=encrypted_key,
            encrypted_secret=encrypted_secret,
            encrypted_passphrase=encrypted_passphrase,
            role=old_key.role,
            status=KeyStatus.ACTIVE,
            description=old_key.description,
            tags=old_key.tags,
            allowed_ips=old_key.allowed_ips,
            allowed_exchanges=old_key.allowed_exchanges,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            rotated_at=now,
            rotation_count=old_key.rotation_count + 1,
            previous_key_id=old_key.id,
        )
        
        await self._log_access(key_id, "rotate", success=True)
        await self._log_access(new_key_id, "create", success=True)
        
        return new_key
    
    async def revoke_key(self, key_id: str, reason: str = "") -> bool:
        """Revoke an API key."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE api_keys 
                SET status = 'revoked', updated_at = NOW(), last_error = $2
                WHERE id = $1 AND status = 'active'
            """, key_id, reason)
        
        if result == "UPDATE 1":
            await self._log_access(key_id, "revoke", success=True, metadata={"reason": reason})
            return True
        return False
    
    async def get_decrypted_credentials(self, key_id: str) -> Optional[Dict[str, str]]:
        """Get decrypted credentials for use in trading."""
        key = await self.get_key(key_id, decrypt=True)
        if not key or not key.can_access("", "", KeyRole.READ):
            return None
        
        # Update usage stats
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE api_keys 
                SET usage_count = usage_count + 1, last_used_at = NOW()
                WHERE id = $1
            """, key_id)
        
        return {
            "api_key": self._encryption.decrypt(key.encrypted_key),
            "api_secret": self._encryption.decrypt(key.encrypted_secret),
            "passphrase": self._encryption.decrypt(key.encrypted_passphrase) if key.encrypted_passphrase else None,
        }
    
    async def record_usage(self, key_id: str, success: bool, error: Optional[str] = None):
        """Record key usage for monitoring."""
        async with self._pool.acquire() as conn:
            if success:
                await conn.execute("""
                    UPDATE api_keys 
                    SET usage_count = usage_count + 1, last_used_at = NOW(), failed_attempts = 0
                    WHERE id = $1
                """, key_id)
            else:
                await conn.execute("""
                    UPDATE api_keys 
                    SET failed_attempts = failed_attempts + 1, last_error = $2
                    WHERE id = $1
                """, key_id, error)
                
                # Check for lockout
                row = await conn.fetchrow("SELECT failed_attempts FROM api_keys WHERE id = $1", key_id)
                if row and row["failed_attempts"] >= self.config.max_failed_attempts:
                    lockout_until = datetime.now(timezone.utc) + timedelta(minutes=self.config.lockout_duration_minutes)
                    await conn.execute("""
                        UPDATE api_keys SET locked_until = $2 WHERE id = $1
                    """, key_id, lockout_until)
        
        await self._log_access(key_id, "use", success=success, metadata={"error": error})
    
    # ============================================================
    # ACCESS LOGGING
    # ============================================================
    
    async def _log_access(
        self,
        key_id: str,
        action: str,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log key access for audit trail."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO key_access_log (key_id, action, success, error_message, metadata)
                VALUES ($1, $2, $3, $4, $5)
            """, key_id, action, success, None, metadata or {})
    
    async def get_access_logs(
        self,
        key_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[KeyAccessLog]:
        """Get access logs for a key."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM key_access_log 
                WHERE key_id = $1
                ORDER BY timestamp DESC
                LIMIT $2 OFFSET $3
            """, key_id, limit, offset)
        
        return [KeyAccessLog(
            id=row["id"],
            key_id=row["key_id"],
            action=row["action"],
            timestamp=row["timestamp"],
            ip=row["ip"],
            user_agent=row["user_agent"],
            success=row["success"],
            error_message=row["error_message"],
            metadata=row["metadata"],
        ) for row in rows]
    
    async def get_expiring_keys(self, days: int = 7) -> List[APIKey]:
        """Get keys expiring within specified days."""
        cutoff = datetime.now(timezone.utc) + timedelta(days=days)
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM api_keys
                WHERE status = 'active' 
                AND expires_at IS NOT NULL
                AND expires_at <= $1
                ORDER BY expires_at ASC
            """, cutoff)
        
        return [self._row_to_key(row) for row in rows]
    
    async def get_rotation_candidates(self, days: int = 30) -> List[APIKey]:
        """Get keys that should be rotated (older than specified days)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM api_keys
                WHERE status = 'active'
                AND (rotated_at IS NULL OR rotated_at < $1)
                AND expires_at > NOW()
                ORDER BY rotated_at ASC NULLS FIRST
            """, cutoff)
        
        return [self._row_to_key(row) for row in rows]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vault statistics."""
        async with self._pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_keys,
                    COUNT(*) FILTER (WHERE status = 'active') as active_keys,
                    COUNT(*) FILTER (WHERE status = 'expired') as expired_keys,
                    COUNT(*) FILTER (WHERE status = 'revoked') as revoked_keys,
                    COUNT(*) FILTER (WHERE expires_at < NOW()) as expired_soon,
                    AVG(usage_count) as avg_usage,
                    MAX(usage_count) as max_usage
                FROM api_keys
            """)
            
            access_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_access,
                    COUNT(*) FILTER (WHERE success = true) as successful_access,
                    COUNT(*) FILTER (WHERE success = false) as failed_access
                FROM key_access_log
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
        
        return {
            "keys": dict(stats) if stats else {},
            "access_24h": dict(access_stats) if access_stats else {},
        }
    
    async def close(self):
        """Close the vault service."""
        await self.stop()


# ============================================================
# FACTORY FUNCTION
# ============================================================

async def create_vault(config: Optional[VaultConfig] = None) -> VaultService:
    """Create and initialize vault service."""
    config = config or VaultConfig.from_env()
    vault = VaultService(config)
    await vault.initialize()
    await vault.start()
    return vault


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "VaultConfig",
    "KeyRole",
    "KeyStatus",
    "APIKey",
    "KeyAccessLog",
    "VaultService",
    "EncryptionService",
    "create_vault",
    "KeyRoleDB",
]