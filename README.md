# Ollama_Local_LLM_Benchmark

A comprehensive benchmarking tool for local Ollama language models, featuring multi-phase testing, GPU telemetry collection, and an interactive HTML dashboard.

## Features

- **Multi-Phase Testing**: Structured test phases for evaluating different model capabilities
- **GPU Telemetry**: Detailed GPU metrics collection (utilization, temperature, power)
- **Interactive Dashboard**: Single-page HTML report with interactive charts
- **Service-Aware**: Automatic detection of ollama-worker services
- **Bilingual Support**: Test cases available in both English and Turkish

## Requirements

- Python 3.8+
- NVIDIA GPU (optional, supports CPU-only mode)
- Ollama with one or more models installed
- Root/sudo access (for initial service detection)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-username/Ollama_Local_LLM_Benchmark.git
cd Ollama_Local_LLM_Benchmark
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the benchmark (requires sudo for service detection):
```bash
sudo python3 ollama_benchmark_setup.py
```

## Usage

The benchmark tool will guide you through:
1. Language selection (English/Turkish/Both)
2. Ollama service selection or manual host entry
3. GPU mode selection (CPU/All GPUs/Specific GPUs)
4. Model selection from available models

After completion, you'll find:
- Test results in `~/ollama_benchmarks/results/`
- Telemetry data in `~/ollama_benchmarks/telemetry/`
- Interactive dashboard in `~/ollama_benchmarks/results/benchmark_master_dashboard_*.html`

## Configuration

Key settings in `ollama_benchmark_setup.py`:
- `GPU_LOG_INTERVAL`: Telemetry collection frequency (default: 2 sec)
- `MAX_RESPONSE_TIME`: Model response timeout (default: 180 sec)
- `TEST_WAIT`: Delay between tests (default: 15 sec)
- `COOLDOWN_AFTER_MODEL`: Cooldown period after each model (default: 15 sec)

## Test Phases

1. **Text Understanding**: Structured data extraction from noisy text
2. **Logical Reasoning**: Data consistency analysis
3. **Analysis & Synthesis**: Multi-criteria evaluation
4. **Dialogue**: Context-aware recommendations
5. **Financial Analysis**: Basic arithmetic and scenario comparison
6. **Code Understanding**: Code analysis and explanation
7. **Long Context**: 4K token context processing
8. **Language Fluency**: Professional text generation

## Dashboard Features

The interactive dashboard provides:
- Per-test-case model comparisons
- Response time analysis
- GPU telemetry overlays (utilization, power, temperature)
- Run selection and filtering

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Publishing this project to GitHub (quick guide)

Aşağıdaki adımlar, projeyi kendi GitHub hesabınızda public bir repository olarak yayınlamanız için gerekli temel komutları içerir. İsterseniz bu adımları manuel olarak takip edebilir veya GitHub CLI (`gh`) yüklüyse otomatikleştirebilirsiniz.

1) Yerel repo oluştur ve ilk commit:

```bash
cd /path/to/Ollama_Local_LLM_Benchmark
git init
git branch -M main
git add .
git commit -m "Initial commit: Ollama_Local_LLM_Benchmark"
```

2) Yeni bir public repository oluşturma ve push (GitHub CLI yüklü ve oturum açıksa):

```bash
# Replace <username> ve <repo-name> ile
gh repo create <username>/<repo-name> --public --source=. --remote=origin --push
```

Alternatif (web arayüzü kullanarak):
 - GitHub'da yeni bir repository oluşturun (Public seçin).
 - Oluşturduğunuz repo için size verilen `git remote add origin ...` komutunu çalıştırın ve `git push -u origin main` ile gönderin.

3) Önemli notlar
 - README dosyasında script adı olarak `ollama_benchmark_setup_v6.py` kullanıldığına dikkat edin.
 - Kişisel veriler ya da gizli anahtar içeren dosyaları `.gitignore` içine eklediğimize dikkat edin (örn. `.env`).
 - `results/`, `telemetry/` ve `logs/` gibi üretim verisi klasörleri zaten `.gitignore` içinde yer alıyor; versiyon kontrolüne yalnızca kaynak kod ve dokümantasyon ekleyin.

4) Yardımcı ipuçları
 - Eğer GitHub Actions ile basit bir CI eklemek isterseniz, ben örnek bir workflow dosyası oluşturabilirim (örn. Python lint ve temel test). Devam etmemi ister misiniz?

---

Eğer isterseniz, bir sonraki adım olarak yerel repo başlatma, commit atma ve GitHub'a push etme komutlarını bende çalıştırmamı sağlayabilirsiniz. Bunun için:
 - Bilgisayarınızda `gh` CLI yüklü ve oturum açmış olmalı; ya da
 - GitHub kullanıcı adınızı ve repo ismini verip, manuel adımları takip etmenize yardımcı olabilirim.

Hangi yolu tercih ediyorsunuz? Otomatik (`gh`) ile devam edeyim mi, yoksa önce lisans ve diğer kontrolleri birlikte gözden mi geçelim?