MNIST APP With Monitoring (Task 1):

    • Modified the mnist api code to include the monitoring of metrics such as the api runtime and total api call counter using prometheus_client.
    • By default, these metrics are available at the address localhost:18000.
    • Downloaded the prometheus-monitoring and node-exporter precompiled binaries from https://prometheus.io/download/ 
    • Modified the prometheus.yml file to include metrics from node-exporter and from the mnist app.
    • By default, node-exporter logs the metrics in the address  localhost:9100 and prometheus has the default address localhost:9090.
    • Installed Grafana by following the steps mention here https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/ 
    • Visualization in Grafana is added from the prometheus data source for the metrics API runtime, total API calls, API memory utilization, API CPU utilization and API network I/O bytes.
    • To start the node-exporter enter the following command from work dir
          cd node_exporter-1.8.0.linux-amd64/ 
          ./node_exporter
          
    • To start prometheus enter the following command from work dir:
          cd prometheus-2.45.5.linux-amd64/
          ./prometheus –config.file=./prometheus.yml

    • Start the mnist api with the command:
          python mnist_api.py Mnist_model.keras

    • Start grafana server with the following command:
          sudo service grafana-server start

    • Grafana dashboard would run at the address localhost:3000(default).


Dockerization of the Mnist API (Task 2):

    • Created the Dockerfile and requirements.txt for dockerization.
    • Built the docker image using the command:
          sudo docker build -t mnist_app .
          
    • Run the docker container using the command:
          sudo docker run -d -p 8000:8000 -p 18000:18000 mnist_app

    • The ports are mapped from the host to the container.
    • Start the grafana dashboard to visualize the metrics and it looks like below.

    • Note that the client ip is 172.17.0.1 which is different from 127.0.0.1 . Hence, we can confirm the app is running in docker container

Running Docker Cluster:
    • To get a cluster of FastAPI servers, multiple docker containers should be run as follows:
          sudo docker run -d -p 8000:8000 -p 18000:18000 --cpus 1 mnist_app
          sudo docker run -d -p 9000:8000 -p 15000:18000 --cpus 1 mnist_app
    • Here, the FastApi server is mapped to different ports to get cluster of servers.
    • Configured the prometheus.yml file to include metrics from different servers and the configuration file is named as prometheus_docker_cluster.yml
    • Start the grafana dashboard to visualize the metrics and it looks like below.

    • Here, the client 172.17.0.1 has two different instances from ports 18000(8000) and 15000(9000) . 
