-- db/init/05_pending_registrations.sql
--
-- Registrations that are not accounts yet: the e-mail must be confirmed first,
-- and municipality registrations additionally need admin approval. Runs on a
-- FRESH pgdata volume; on existing volumes the backend's startup
-- Base.metadata.create_all() creates this table instead — keep in sync with
-- backend/models_auth.py::PendingRegistration.

CREATE TABLE IF NOT EXISTS pending_registrations (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at       TIMESTAMPTZ DEFAULT now(),

    username         VARCHAR(40)  NOT NULL UNIQUE,
    email            VARCHAR(120) NOT NULL UNIQUE,
    full_name        VARCHAR(120),
    password_hash    VARCHAR(300) NOT NULL,
    role             VARCHAR(16)  NOT NULL DEFAULT 'user',   -- user | municipality
    city             VARCHAR(80),

    email_code       VARCHAR(12),
    code_expires_at  TIMESTAMPTZ,
    email_verified   BOOLEAN NOT NULL DEFAULT FALSE,
    status           VARCHAR(20) NOT NULL DEFAULT 'awaiting_email'
                     -- awaiting_email | awaiting_approval
);
