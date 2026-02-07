from locust import HttpUser, between, task


class RecommenderUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def health(self):
        self.client.get("/health")

    @task
    def theme(self):
        self.client.post(
            "/recommendations/theme",
            json={"theme": "egypt", "k": 5, "min_similarity": 0.0},
        )
