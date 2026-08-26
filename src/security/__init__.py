"""
QuantAI Security Package

Vault service for secure API key management with PostgreSQL + Fernet encryption.
"""

from .vault import (
    VaultConfig,
    KeyRole,
    KeyStatus,
    APIKey,
    KeyAccessLog,
    VaultService,
    EncryptionService,
    create_vault,
    KeyRoleDB,
)

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