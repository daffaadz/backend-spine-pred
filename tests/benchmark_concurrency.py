import time
import asyncio
import httpx
from app.main import app

async def benchmark_concurrent_users(n_users: int = 10, requests_per_user: int = 5):
    transport = httpx.ASGITransport(app=app)
    payload = {
        "pelvic_incidence": 52.4,
        "pelvic_tilt": 14.2,
        "lumbar_lordosis_angle": 48.1,
        "sacral_slope": 38.2,
        "pelvic_radius": 118.5,
        "degree_spondylolisthesis": 4.3,
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Warmup
        await client.post("/api/predict", json=payload)

        async def worker(user_id: int):
            latencies = []
            for _ in range(requests_per_user):
                t0 = time.perf_counter()
                resp = await client.post("/api/predict", json=payload)
                dt = (time.perf_counter() - t0) * 1000.0
                assert resp.status_code == 200
                latencies.append(dt)
            return latencies

        t_start = time.perf_counter()
        tasks = [worker(i) for i in range(n_users)]
        results = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - t_start) * 1000.0

    all_latencies = [lat for user_lats in results for lat in user_lats]
    total_requests = len(all_latencies)
    avg_latency = sum(all_latencies) / total_requests
    p95_latency = sorted(all_latencies)[int(0.95 * total_requests)]
    rps = total_requests / (total_time / 1000.0)

    print("\n=== BENCHMARK HASIL KONKURENSI (10 USERS) ===")
    print(f"Total Concurrent Users : {n_users}")
    print(f"Total Requests Executed: {total_requests}")
    print(f"Total Duration         : {total_time:.2f} ms")
    print(f"Throughput             : {rps:.2f} req/s")
    print(f"Rata-rata Latensi      : {avg_latency:.2f} ms")
    print(f"P95 Latensi            : {p95_latency:.2f} ms")
    print("============================================\n")

if __name__ == "__main__":
    from app.services.ml_service import ml_service
    ml_service.load_model()
    asyncio.run(benchmark_concurrent_users(n_users=10, requests_per_user=10))
