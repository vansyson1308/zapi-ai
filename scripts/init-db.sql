-- 2api.ai Database Schema
-- PostgreSQL 16+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- TENANTS (Customers)
-- ============================================================
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    plan VARCHAR(50) DEFAULT 'free' CHECK (plan IN ('free', 'starter', 'pro', 'enterprise')),
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for email lookups
CREATE INDEX idx_tenants_email ON tenants(email);
CREATE INDEX idx_tenants_plan ON tenants(plan);

-- ============================================================
-- API KEYS
-- ============================================================
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    key_hash VARCHAR(64) NOT NULL,  -- SHA-256 hash of the full key
    key_prefix VARCHAR(12) NOT NULL, -- "2api_" + first 7 chars for identification
    name VARCHAR(255) DEFAULT 'Default Key',
    permissions JSONB DEFAULT '["*"]',
    rate_limit_per_minute INTEGER DEFAULT 60,
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- ============================================================
-- PROVIDER CONFIGURATIONS
-- ============================================================
CREATE TABLE provider_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL CHECK (provider IN ('openai', 'anthropic', 'google')),
    api_key_encrypted TEXT,  -- Encrypted with tenant-specific key
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tenant_id, provider)
);

CREATE INDEX idx_provider_configs_tenant ON provider_configs(tenant_id);

-- ============================================================
-- USAGE RECORDS
-- ============================================================
CREATE TABLE usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    
    -- Request details
    request_id VARCHAR(64) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    operation VARCHAR(50) NOT NULL CHECK (operation IN ('chat', 'embedding', 'image')),
    
    -- Token usage
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    
    -- Cost
    cost_usd DECIMAL(10, 6) DEFAULT 0,
    
    -- Performance
    latency_ms INTEGER,
    
    -- Status
    status VARCHAR(20) CHECK (status IN ('success', 'error', 'timeout', 'rate_limited')),
    error_code VARCHAR(50),
    error_message TEXT,
    
    -- Routing
    routing_strategy VARCHAR(50),
    fallback_used BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_usage_tenant_created ON usage_records(tenant_id, created_at DESC);
CREATE INDEX idx_usage_provider ON usage_records(provider, created_at DESC);
CREATE INDEX idx_usage_model ON usage_records(model, created_at DESC);
CREATE INDEX idx_usage_status ON usage_records(status);
CREATE INDEX idx_usage_created ON usage_records(created_at DESC);

-- Partition by month for better performance (optional for large scale)
-- CREATE TABLE usage_records_2024_01 PARTITION OF usage_records
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ============================================================
-- PROVIDER HEALTH
-- ============================================================
CREATE TABLE provider_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider VARCHAR(50) NOT NULL UNIQUE,
    is_healthy BOOLEAN DEFAULT TRUE,
    avg_latency_ms INTEGER,
    error_rate DECIMAL(5, 4) DEFAULT 0,
    last_error TEXT,
    last_check_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Initialize provider health records
INSERT INTO provider_health (provider, is_healthy) VALUES
    ('openai', TRUE),
    ('anthropic', TRUE),
    ('google', TRUE);

-- ============================================================
-- ROUTING POLICIES
-- ============================================================
CREATE TABLE routing_policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    
    -- Policy configuration
    strategy VARCHAR(50) DEFAULT 'cost' CHECK (strategy IN ('cost', 'latency', 'quality', 'custom')),
    fallback_chain TEXT[] DEFAULT '{}',
    
    -- Constraints
    max_latency_ms INTEGER,
    max_cost_per_request DECIMAL(10, 4),
    
    -- Rules (for custom strategy)
    rules JSONB DEFAULT '[]',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_routing_policies_tenant ON routing_policies(tenant_id);

-- ============================================================
-- BILLING RECORDS (for token-based billing)
-- ============================================================
CREATE TABLE billing_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- Billing period
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- Usage summary
    total_requests INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    
    -- Cost breakdown
    total_cost_usd DECIMAL(12, 4) DEFAULT 0,
    cost_by_provider JSONB DEFAULT '{}',  -- {"openai": 10.50, "anthropic": 5.25}
    cost_by_model JSONB DEFAULT '{}',     -- {"gpt-4o": 8.00, "claude-3-5-sonnet": 7.75}
    
    -- Billing status
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'invoiced', 'paid', 'failed')),
    invoice_id VARCHAR(100),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tenant_id, period_start, period_end)
);

CREATE INDEX idx_billing_tenant ON billing_records(tenant_id);
CREATE INDEX idx_billing_period ON billing_records(period_start, period_end);

-- ============================================================
-- VIEWS for common queries
-- ============================================================

-- Daily usage summary
CREATE OR REPLACE VIEW daily_usage_summary AS
SELECT
    tenant_id,
    DATE(created_at) as date,
    provider,
    model,
    COUNT(*) as request_count,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    SUM(total_tokens) as total_tokens,
    SUM(cost_usd) as total_cost,
    AVG(latency_ms)::INTEGER as avg_latency_ms,
    COUNT(*) FILTER (WHERE status = 'error') as error_count
FROM usage_records
GROUP BY tenant_id, DATE(created_at), provider, model;

-- Provider health overview
CREATE OR REPLACE VIEW provider_overview AS
SELECT
    p.provider,
    p.is_healthy,
    p.avg_latency_ms,
    p.error_rate,
    p.last_check_at,
    COALESCE(u.requests_last_hour, 0) as requests_last_hour,
    COALESCE(u.errors_last_hour, 0) as errors_last_hour
FROM provider_health p
LEFT JOIN (
    SELECT
        provider,
        COUNT(*) as requests_last_hour,
        COUNT(*) FILTER (WHERE status = 'error') as errors_last_hour
    FROM usage_records
    WHERE created_at > NOW() - INTERVAL '1 hour'
    GROUP BY provider
) u ON p.provider = u.provider;

-- ============================================================
-- FUNCTIONS
-- ============================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables
CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_provider_configs_updated_at
    BEFORE UPDATE ON provider_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_routing_policies_updated_at
    BEFORE UPDATE ON routing_policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_billing_records_updated_at
    BEFORE UPDATE ON billing_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- SAMPLE DATA (for development)
-- ============================================================

-- Create a test tenant
INSERT INTO tenants (id, name, email, plan) VALUES
    ('00000000-0000-0000-0000-000000000001', 'Test User', 'test@example.com', 'pro');

-- Create a test API key (key: 2api_test_key_12345, hash is SHA-256 of this)
INSERT INTO api_keys (tenant_id, key_hash, key_prefix, name) VALUES
    ('00000000-0000-0000-0000-000000000001', 
     'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 
     '2api_test_k',
     'Development Key');

-- Create default routing policy
INSERT INTO routing_policies (tenant_id, name, is_default, strategy, fallback_chain) VALUES
    ('00000000-0000-0000-0000-000000000001', 
     'Default Policy',
     TRUE,
     'cost',
     ARRAY['openai/gpt-4o-mini', 'anthropic/claude-3-haiku', 'google/gemini-1.5-flash']);

COMMENT ON TABLE tenants IS '2api.ai customers/organizations';
COMMENT ON TABLE api_keys IS 'API keys for authentication';
COMMENT ON TABLE usage_records IS 'Request logs for billing and analytics';
COMMENT ON TABLE provider_health IS 'Real-time health status of AI providers';
COMMENT ON TABLE routing_policies IS 'Custom routing rules per tenant';
COMMENT ON TABLE billing_records IS 'Monthly billing summaries';
