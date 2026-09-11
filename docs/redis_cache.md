# Redis Response Cache

## Purpose

Redis is an optional shared response cache for the FastAPI serving layer. It
reduces repeated dataframe filtering, model inference, report assembly, and
JSON serialization for identical requests. It does not store source datasets,
training data, or model artifacts.

Cached endpoints are:

| Endpoint | Default TTL |
| --- | ---: |
| `/recommendations` | 21,600 seconds (6 hours) |
| `/forecasts/short-term` | 3,600 seconds (1 hour) |
| `/forecasts/long-term` | 3,600 seconds (1 hour) |
| `/players/scouting-report` | 3,600 seconds (1 hour) |

Every key contains a namespace, runtime version, and SHA-256 request digest:

```text
nba-scout:<runtime-version>:<endpoint-namespace>:<request-hash>
```

The request hash includes every schema field. Two requests with different
seasons, presets, horizons, filters, or requested tasks cannot share a cached
response.

## Data Versioning

Set `DATA_VERSION` to the immutable serving snapshot identifier, such as a data
release date or DVC revision:

```text
DATA_VERSION=gold-2024-25-v1
```

When it is omitted, the API derives a version from serving data and artifact
file metadata. An explicit version is preferred in ECS because it is visible
in deployment configuration and can be tied to a nightly pipeline output.

After a new Gold snapshot or model artifact is deployed, update
`DATA_VERSION`. Old keys expire through TTL and are never read by the new
runtime.

## Failure Behavior

Redis is not required for correctness:

1. If `REDIS_URL` is absent, the API uses a no-op cache.
2. If Redis cannot be reached during startup, FastAPI still starts and reports
   the cache as `unavailable`.
3. If Redis fails after startup, the request is computed normally and a
   30-second circuit breaker prevents repeated connection timeouts.
4. API responses are cached only after successful computation.

Inspect cache state through:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/metadata
```

The response includes cache status, hits, misses, hit rate, errors, and circuit
breaker bypasses.

## Local Usage

Run Redis and the API together:

```bash
docker compose up --build
```

Then send the same request twice and inspect `/health`. The first request is a
miss and the second is a hit.

To run only Redis while starting Uvicorn from the host:

```bash
docker compose up -d redis
export REDIS_URL=redis://localhost:6379/0
export DATA_VERSION=local-development
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## AWS ECS And ElastiCache

The recommended cloud topology is:

```text
Internet or ALB
    -> ECS Fargate service security group
        -> FastAPI tasks
            -> ElastiCache security group on TCP 6379
```

### 1. Create the cache

Create an Amazon ElastiCache Redis-compatible cache in the same VPC as the ECS
service. Select private subnets that are routable from the ECS task subnets.
Enable in-transit encryption and authentication for a production deployment.

### 2. Use dedicated security groups

Create an ECS task security group, for example `nba-scout-api-sg`, and a cache
security group, for example `nba-scout-cache-sg`.

Configure the cache security group inbound rule as:

```text
Type: Custom TCP
Port: 6379
Source: nba-scout-api-sg
```

Do not use `0.0.0.0/0` and do not assign a public endpoint to Redis. The ECS
security group needs outbound access to the cache security group.

### 3. Store the connection URL as a secret

Store the connection URL in AWS Secrets Manager rather than placing credentials
in GitHub or the task-definition environment block. Depending on the selected
ElastiCache authentication mode, the URL resembles:

```text
rediss://default:<token>@<cache-endpoint>:6379/0
```

Reference that secret from the `nba-scout-api` container as `REDIS_URL`. Ensure
the ECS task execution role can retrieve the secret.

### 4. Add non-secret task environment values

Add these values to the container definition:

```text
DATA_VERSION=gold-2024-25-v1
RECOMMENDATION_CACHE_TTL_SECONDS=21600
SCOUTING_REPORT_CACHE_TTL_SECONDS=3600
FORECAST_CACHE_TTL_SECONDS=3600
REDIS_CONNECT_TIMEOUT_SECONDS=0.5
REDIS_SOCKET_TIMEOUT_SECONDS=0.5
REDIS_FAILURE_COOLDOWN_SECONDS=30
```

For a nightly data release, update `DATA_VERSION` to the new snapshot ID when
deploying or restarting the service.

### 5. Deploy and verify

Register a new task-definition revision and update the ECS service. Verify:

```text
GET /health -> cache.backend = redis
GET /health -> cache.status = ready
```

Send an identical recommendation twice and verify that `misses` increases on
the first request and `hits` increases on the second.

## Operational Notes

- Use an eviction policy suitable for ephemeral response data, such as
  `allkeys-lru`.
- Cache availability must not control ECS container health; FastAPI remains
  healthy when Redis is degraded.
- Monitor hit rate, errors, memory usage, evictions, and endpoint latency.
- A shared managed cache becomes most useful when the ECS service runs more
  than one task. A process-local cache cannot share results across replicas.
