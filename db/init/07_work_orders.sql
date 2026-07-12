-- db/init/07_work_orders.sql
--
-- Municipality repair workflow: work orders group detections into repair jobs
-- (status flow: open → scheduled → in_progress → repaired → verified, or
-- cancelled). Runs on a FRESH pgdata volume; existing volumes get the tables
-- from Base.metadata.create_all() — keep in sync with backend/models_work.py.

CREATE TABLE IF NOT EXISTS work_orders (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at         TIMESTAMPTZ DEFAULT now(),
    updated_at         TIMESTAMPTZ DEFAULT now(),

    city               VARCHAR(80) NOT NULL,
    title              VARCHAR(120) NOT NULL,
    status             VARCHAR(16) NOT NULL DEFAULT 'open',

    crew_name          VARCHAR(80),
    scheduled_for      DATE,
    due_date           DATE,
    cost_estimate_ron  FLOAT,
    cost_actual_ron    FLOAT,
    notes              TEXT,

    created_by         UUID REFERENCES users(id) ON DELETE SET NULL,
    completed_at       TIMESTAMPTZ,
    verified_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_work_orders_city_status ON work_orders (city, status);

CREATE TABLE IF NOT EXISTS work_order_items (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    work_order_id  UUID NOT NULL REFERENCES work_orders(id) ON DELETE CASCADE,
    detection_id   UUID NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    sort_order     INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT uq_woi_pair UNIQUE (work_order_id, detection_id)
);

CREATE INDEX IF NOT EXISTS idx_woi_order     ON work_order_items (work_order_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_woi_detection ON work_order_items (detection_id);
