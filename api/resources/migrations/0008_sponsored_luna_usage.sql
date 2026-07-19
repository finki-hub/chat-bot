CREATE TABLE IF NOT EXISTS sponsored_user_usage (
    user_id UUID NOT NULL,
    usage_date DATE NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0 CHECK (request_count >= 0),
    PRIMARY KEY (user_id, usage_date)
);

CREATE TABLE IF NOT EXISTS sponsored_global_usage (
    usage_date DATE NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0 CHECK (request_count >= 0),
    PRIMARY KEY (usage_date)
);

CREATE TABLE IF NOT EXISTS sponsored_request_leases (
    user_id UUID NOT NULL,
    request_id UUID NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (user_id)
);
