-- Runs once, the first time the PostgreSQL container is created.

-- Schema for the app's own tables (datasets, messages, logs ...)
CREATE SCHEMA IF NOT EXISTS app;

-- Schema where every uploaded dataset becomes its own table
CREATE SCHEMA IF NOT EXISTS user_data;

-- Read-only role: AI-generated SQL is executed as this user (development password!)
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'insightflow_readonly') THEN
    CREATE ROLE insightflow_readonly LOGIN PASSWORD 'readonly_dev_password';
  END IF;
END $$;

GRANT CONNECT ON DATABASE insightflow TO insightflow_readonly;
GRANT USAGE ON SCHEMA user_data TO insightflow_readonly;

-- Every table the admin creates in user_data can be read (but never changed) by the read-only role
ALTER DEFAULT PRIVILEGES IN SCHEMA user_data GRANT SELECT ON TABLES TO insightflow_readonly;

-- No access at all to the app's own tables
REVOKE ALL ON SCHEMA app FROM insightflow_readonly;
-- The role also must not be able to create objects in the public schema
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
