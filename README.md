# Evaluating Memanto for Tool Using AI agents

### 1. Clone the Memanto benchmark repo

```bash
git clone https://github.com/moorcheh-ai/memanto-benchmark.git
```

### 2. Clone DolphinBench alongside it

```bash
git clone https://github.com/mem0ai/dolphinbench.git
```

### 3. Enter the DolphinBench checkout

```bash
cd dolphinbench
```

### 4. Create a Python virtual environment

```bash
python3 -m venv .venv
```

### 5. Activate the environment

```bash
source .venv/bin/activate
```

### 6. Install DolphinBench dependencies

```bash
python -m pip install -r requirements.txt
```

### 7. Install Memanto and its Hermes integration

```bash
python -m pip install memanto python-dotenv hermes-memanto
```

### 8. Copy the adaptive harness

```bash
cp ../memanto-benchmark/my_harness_adaptive.py .
```

### 9. Copy the benchmark configuration

```bash
cp ../memanto-benchmark/config.yaml .
```

### 10. Install Hermes if needed

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### 11. Install the Memanto plugin in Hermes

```bash
hermes-memanto-install
```

### 12. Activate Memanto as the memory provider

```bash
hermes memory setup memanto
```

### 13. Verify the active memory provider

```bash
hermes memory status
```

### 14. Set your OpenAI key

```bash
export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
```

### 15. Set your Moorcheh key

```bash
export MOORCHEH_API_KEY='YOUR_MOORCHEH_API_KEY'
```

### 16. Edit the existing configuration

Set the Hermes Python path and output directory. For all 600 tasks, select Alex, Morgan, and Riley with task IDs `001`–`200`. The adaptive settings are `top_k: 50` and `recall_min_similarity: 0.1`.

```bash
nano config.yaml
```

### 17. Check that Hermes loads the Memanto plugin

```bash
python my_harness_adaptive.py runtime-check --config config.yaml
```

### 18. Prepare the benchmark run

```bash
python my_harness_adaptive.py prepare --config config.yaml
```

### 19. Ingest the persona histories

Each persona uses a separate memory namespace. Skip this step if `memory_run_dir` points to completed ingestion for the selected personas.

```bash
python -u my_harness_adaptive.py ingest --config config.yaml --confirm-paid-calls
```

### 20. Run the evaluation

The harness connects DolphinBench’s simulated apps to Hermes automatically.

```bash
python -u my_harness_adaptive.py evaluate --config config.yaml --confirm-paid-calls
```

### 21. Print the results

The summary reports accuracy, median task latency, and recorded agent cost.

```bash
python my_harness_adaptive.py summary --config config.yaml
```
