#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import time
import json
import csv
import re
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import pwd  # moved here for consistent imports

# ---------- CONFIG ----------
SSD_READ_MBPS = 500.0
GPU_STABILIZE_SEC = 15
PING_RETRIES = 3
PING_RETRY_WAIT = 6
MAX_RESPONSE_TIME = 180
MAX_OUTPUT_CHARS = 5000
TEST_WAIT = 15
COOLDOWN_AFTER_MODEL = 15
GPU_LOG_INTERVAL = 2  # sec
PY_DEPS = ["pandas", "nvidia-ml-py3", "plotly", "rich"]
# ----------------------------

# -------- paths dynamic by user ----------
_sudo_user = os.environ.get("SUDO_USER")
if _sudo_user:
    try:
        USER_HOME = Path(pwd.getpwnam(_sudo_user).pw_dir)
    except Exception:
        USER_HOME = Path(os.path.expanduser(f"~{_sudo_user}")) if _sudo_user else Path.home()
else:
    USER_HOME = Path.home()
BASE = USER_HOME / "ollama_benchmarks"
TEST_DIR = BASE / "test_sets"
RESULTS_DIR = BASE / "results"
LOGS_DIR = BASE / "logs"
TELEMETRY_DIR = BASE / "telemetry"
VENV_DIR = BASE / "venv"
MODEL_META = BASE / "model_sizes.json"
SELECTED_MODELS_FILE = BASE / "selected_models.json"

# --------- helpers: rich console ----------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    console = Console()
except Exception:
    console = None

def cprint(text, style=None, end="\n"):
    if console:
        console.print(text, style=style, end=end)
    else:
        print(text, end=end)

# --------- subprocess helper -------------
def run_cmd(cmd, timeout=None, env=None, capture=True):
    """Run shell command and return (stdout, stderr, rc)."""
    try:
        if capture:
            p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=timeout)
            return p.stdout.strip(), p.stderr.strip(), p.returncode
        else:
            rc = subprocess.call(cmd, shell=True, env=env)
            return "", "", rc
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", 124

# --------- backup helper -----------------
def backup_if_exists(path: Path):
    """Back up a file or directory if it exists."""
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = path.parent / f"backup_{path.name}_{ts}"
        if path.is_dir():
            shutil.copytree(path, dst)
        else:
            shutil.copy2(path, dst)
        cprint(f"[yellow]Backed up:[/yellow] {path} -> {dst}")

def prepare_dirs():
    """Prepare required directories."""
    for p in [BASE, TEST_DIR, RESULTS_DIR, LOGS_DIR, TELEMETRY_DIR, VENV_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    cprint(f"[green]Folders ready: {BASE}[/green]")

# ---------- write test files -------------
TEST_FILES_CONTENT = {
        'tr': {
                "phase1_text_understanding.txt": """🔹 PHASE 1 — Ham Veriden Yapılandırılmış Bilgi Çıkarma (Gürültülü Metin)

Amaç: Modelin, bozuk formatlı, eksik veya tutarsız bilgiler içeren bir metinden doğru veri ayıklama, normalize etme ve JSON formatında çıktı verme becerisini ölçmek.

Prompt: 
Aşağıdaki metin bir e-ticaret sitesinden alınmıştır. Metinde reklam dili, gereksiz ifadeler ve hatalı yazımlar bulunmaktadır.  
Metindeki bilgileri analiz et, aşağıdaki JSON şemasına **birebir uyan** bir çıktı oluştur.  
Eksik bilgiler için null değeri kullan.  
Sadece JSON çıktısı ver.

JSON Şeması:
{
    "urun_adi": "string",
    "marka": "string",
    "model_kodu": "string",
    "kategori": "string",
    "fiyat": {
        "deger": "float",
        "para_birimi": "string"
    },
    "teknik_ozellikler": {
        "islemci": "string",
        "ram_gb": "integer",
        "depolama_gb": "integer",
        "ekran_boyutu_inc": "float",
        "cozunurluk": "string",
        "kamera_mp": "integer",
        "batarya_suresi_saat": "integer"
    },
    "one_cikan_ozellikler": ["string"],
    "kullanici_yorumu_ozeti": "string"
}

Ham Metin:
“YENİ! GALAXY S25-Ultraaa 😍 (model: smg998ds). 6.8inc AMOLED ekraan—göz kamaştırır!  
Çip: SnapDragon 9 Gen2, tam 16GB RAM💥, 512 depolama gb.  
Fiyat şu an sadece ₺49,999 (bazı yerlerde 52.000TL deniyor).  
Kamera: 200mp (arka), batarya: 5000mAh—yaklaşık 20 saat video!  
Ekstralar: IP68 koruma, hızlı/kablosuz şarj, S-Pen desteği.  
Bir kullanıcı diyor ki: “Bataryası efsane ama biraz ağır.”  
Bu ürün akıllı telefon kategorisindedir.”""",

                "phase2_logical_reasoning.txt": """🔹 PHASE 2 — Mantık Yürütme ve Veri Tutarlılığı Analizi

Amaç: Modelin, eksik veya çelişkili verileri tespit edip, bunlara yönelik "mantıklı araştırma görevleri" tanımlama kabiliyetini ölçmek.

Prompt:
Bir "Veri Bütünlüğü Denetçisi" gibi davran.  
Aşağıdaki ürün kaydını analiz et.

1 Eksik (null) değerleri tespit et.  
2 kayit_tarihi ve guncel_tarih değerlerini karşılaştırarak, güncelliğini yitirmiş olabilecek alanları belirle.  
3 Aşağıdaki JSON formatında, yapılması gereken web araştırma görevlerini oluştur.  
     Her görev, gorev_tipi ('EKSİK_BİLGİ' veya 'GÜNCELLEME') ve arastirma_sorgusu alanlarını içermelidir.  
     Sadece bu JSON dizisini ver.

JSON Formatı:
[
    {"gorev_tipi": "EKSİK_BİLGİ", "arastirma_sorgusu": "string"},
    {"gorev_tipi": "GÜNCELLEME", "arastirma_sorgusu": "string"}
]

Veritabanı Kaydı:
{
    "urun_id": "laptop_aurora_2023",
    "urun_adi": "AuroraBook Air 13",
    "marka": "TechNova",
    "kayit_tarihi": "2023-04-10",
    "guncel_tarih": "2025-10-22",
    "fiyat_tl": 28999,
    "teknik_ozellikler": {
        "islemci": "Intel Core i5-1230U",
        "ram_gb": 8,
        "depolama_gb": 512,
        "ekran_karti": null,
        "agirlik_kg": 1.3
    },
    "rakip_modeller": []
}""",

                "phase3_analysis_synthesis.txt": """🔹 PHASE 3 — Analiz, Sentez ve Karşılaştırmalı Değerlendirme

Amaç: Modelin, kullanıcı önceliklerine göre çok kriterli değerlendirme ve puanlama yapma yeteneğini ölçmek.

Prompt:
Bir "Ürün Analisti" gibi davran.  
Aşağıdaki kullanıcı profilini dikkate alarak iki akıllı telefonu değerlendir:

Kullanıcı Profili:
"Sık seyahat eden bir profesyonel. Öncelikler: uzun pil ömrü (%40), yüksek kaliteli kamera (%35), hafif tasarım (%25)."

Görevlerin:
1 Her iki telefonu profil önceliklerine göre analiz et.  
2 Her biri için 3 artı ve 3 eksi yön belirt.  
3 10 üzerinden “Uygunluk Puanı” hesapla (öncelik ağırlıklarını dikkate al).  
4 Sonuçta hangi telefonun daha uygun olduğunu belirt.

Ürün Verileri:
[
    {
        "urun_adi": "Pixel Pro 9",
        "pil_omru_saat": 22,
        "kamera_puani_dxo": 155,
        "agirlik_gram": 185,
        "ozellikler": ["Yapay zeka destekli kamera", "Anında çeviri", "Temiz Android arayüzü"]
    },
    {
        "urun_adi": "ZenPhone Max",
        "pil_omru_saat": 30,
        "kamera_puani_dxo": 140,
        "agirlik_gram": 220,
        "ozellikler": ["Devasa batarya", "Oyun performansı modu", "Dayanıklı kasa"]
    }
]""",

                "phase4_dialogue_recommendation.txt": """🔹 PHASE 4 — Diyalog ve Bağlam-İçi Tavsiye (RAG Mantığı Testi)

Amaç: Modelin, kullanıcı niyetini sezme, bağlam içi verileri kullanma ve en uygun öneriyi doğal diyalog formunda sunma yeteneğini ölçmek.

Prompt:
Bir "Akıllı Satış Danışmanı Chatbot" gibi davran.  
Kullanıcının sorusunu analiz et ve BAĞLAM içindeki ürünlerden en uygun olanı öner.  
Kullanıcının mesajında gizli ipuçlarını da dikkate al (örneğin "taşınabilirlik" veya "ısı yönetimi" gibi).  
Sadece BAĞLAM içindeki bilgileri kullan.  

Kullanıcı Sorusu:
“Merhaba, bütçem 1500 dolar civarı. Hem oyun oynuyorum hem de sık seyahat ediyorum, şarjımın uzun gitmesi benim için önemli ama çok ağır cihazlardan da hoşlanmıyorum. Hangisini önerirsiniz?”

BAĞLAM:
- Ürün: **Phantom X Gamer** | Fiyat: 1450 USD | Pil: 5500 mAh (Oyun: 10 saat) | Ağırlık: 240g | Soğutma: Buhar odası | Ekran: 144Hz  
- Ürün: **Elite Slim V2** | Fiyat: 1499 USD | Pil: 4500 mAh (Oyun: 6 saat) | Ağırlık: 175g | Tasarım: Ultra ince | Ekran: 90Hz  
- Ürün: **PowerHouse One** | Fiyat: 1300 USD | Pil: 6000 mAh (Oyun: 12 saat) | Ağırlık: 230g | Soğutma: Standart | Ekran: 120Hz""",

                "phase5_financial_analysis.txt": """🔹 PHASE 5 — Finansal Karar Analizi (Basitleştirilmiş Versiyon)

Amaç: Modelin, basit matematiksel hesap ve senaryo karşılaştırması yapabilme becerisini ölçmek.

Prompt:
Bir yatırımcı, 1.000.000 TL birikimiyle bir bahçenin yarısını almak istiyor.  
Toplam 1000 m² olan bahçenin m² fiyatı 2.300 TL'dir.  
Yıllık faiz oranı %50 (aylık bileşik faiz oranı ≈ 3,45%).

Satıcı 3 teklif sunuyor:

1 Peşin ödeme: 1.150.000 TL hemen.  
2 6 ay vadeli tek ödeme: 1.250.000 TL (6. ayda tek seferde).  
3 6 taksitli ödeme: toplam 1.200.000 TL, her ay eşit ödemeli.

Yatırımcının elindeki parayı banka faizinde değerlendirip değerlendirmenin mantıklı olup olmadığını analiz et.  
6 ay sonunda her senaryodaki **net nakit farkını** hesapla.  
Sonucu JSON formatında ver.

JSON Formatı:
{
    "senaryolar": {
        "pesin": {"net_nakit_6ay": "float"},
        "taksitli": {"net_nakit_6ay": "float"},
        "ertelenmis": {"net_nakit_6ay": "float"}
    },
    "en_karlı_secenek": "string"
}""",

                "phase6_code_understanding.txt": """🔹 PHASE 6 — Kod Anlama ve Açıklama

Amaç: Modelin, kısa bir kod parçasının ne yaptığını doğru anlamlandırma ve Türkçe olarak sade bir şekilde açıklama becerisini ölçmek.

Prompt:
Aşağıdaki Python kodunun ne yaptığını sade bir dille açıkla.
Sadece işlevsel özeti ver. Kodun tek tek satırlarını açıklama.

Kod:
def analiz(dizi):
        toplam = sum(dizi)
        ort = toplam / len(dizi)
        farklar = [(x - ort) ** 2 for x in dizi]
        varyans = sum(farklar) / len(dizi)
        return {"ortalama": ort, "standart_sapma": varyans ** 0.5}

print(analiz([10, 12, 8, 15, 9]))""",

                "phase7_long_context.txt": """🔹 PHASE 7 — Uzun Bağlam Anlama (4K Token İçin Optimize)

Amaç: Modelin, uzun metinlerde önemli detayları koruyarak doğru özet çıkarma kabiliyetini test etmek.

Prompt:
Aşağıdaki uzun metni dikkatlice oku.  
Metin 4K token sınırlı modellerin kapasitesini zorlamadan analiz yapabilecek uzunluktadır.  
Metnin sonunda verilen görevleri yerine getir.  
Çıktıyı JSON formatında ver.

Görevler:
1 Metindeki en önemli 3 temayı belirle.  
2 Bu temalardan birine ilişkin öneri sun.  
3 Metnin genel tonunu (iyimser, eleştirel, tarafsız vb.) belirt.

JSON Formatı:
{
    "temalar": ["string"],
    "onerilen_gelisim": "string",
    "genel_ton": "string"
}

Metin:
Küresel enerji dönüşümü, yalnızca fosil yakıtlardan yenilenebilir kaynaklara geçişi değil, aynı zamanda enerji depolama, akıllı şebekeler ve verimlilik teknolojilerinin bütünleşmesini de kapsıyor.  
Son beş yılda, özellikle güneş ve rüzgar enerjisinde maliyetler %60'tan fazla düştü. Ancak, enerji depolama teknolojilerinde hâlâ kritik darboğazlar mevcut.  
Lityum-iyon pillerin üretimi, çevresel maliyetler ve arz zinciri sorunları nedeniyle sürdürülebilirlik açısından eleştiriliyor.  
Araştırmacılar, katı hal batarya ve hidrojen depolama çözümlerine yöneliyor.  
Bununla birlikte, bazı ülkelerde politika tutarsızlıkları ve altyapı eksiklikleri, yenilenebilir enerji hedeflerine ulaşmayı geciktiriyor.  
Uluslararası Enerji Ajansı, 2030 yılına kadar bu alanda küresel yatırımın üç katına çıkması gerektiğini belirtiyor.""",

                "phase8_turkish_fluency.txt": """🔹 PHASE 8 — Türkçe Dil Akıcılığı ve Üslup Testi

Amaç: Modelin, profesyonel bir tanıtım metni üretme, dil akıcılığı ve biçemsel bütünlük kurma becerisini ölçmek.

Prompt:
Bir teknoloji dergisi için profesyonel bir tanıtım yazısı oluştur.  
Konu: "Yapay zekânın günlük hayatta görünmez ama etkili kullanımları."  
Metin 3 paragraf olmalı.  
Resmî ama akıcı bir dil kullan.  
Anlatım tutarlı, mantıksal geçişleri düzgün ve kurumsal tonda olsun."""
        },
        'en': {
                "phase1_structured_extraction.txt": """🔹 PHASE 1 — Structured Data Extraction from Noisy Text

Objective: Measure the model's ability to extract, normalize and output precise JSON from a noisy/messy product description.

Prompt:
The following text is taken from an e-commerce listing and contains marketing language, typos and inconsistent formatting. Extract the data to exactly match the JSON schema below. Use null for missing fields. Return only the JSON.

JSON Schema:
{
    \"product_name\": \"string\",
    \"brand\": \"string\",
    \"model_code\": \"string\",
    \"category\": \"string\",
    \"price\": {
        \"value\": \"float\",
        \"currency\": \"string\"
    },
    \"technical_specifications\": {
        \"processor\": \"string\",
        \"ram_gb\": \"integer\",
        \"storage_gb\": \"integer\",
        \"screen_size_inch\": \"float\",
        \"resolution\": \"string\",
        \"camera_mp\": \"integer\",
        \"battery_life_hours\": \"integer\"
    },
    \"highlighted_features\": [\"string\"],
    \"user_review_summary\": \"string\"
}

Raw Text: "The Galaxy S25 Ultra is here for tech enthusiasts! Developed by Samsung, this device (model SM-G998B/DS) dazzles with its massive 6.8-inch Dynamic AMOLED screen. It's powered by the Snapdragon 9 Gen 2 processor and comes with a full 16 GB of RAM. The phone, offering 512 GB of storage, is priced at 49,999 TRY. Take professional photos with its 200 Megapixel rear camera. Thanks to its 5000mAh battery, it offers 'up to 20 hours of video playback'. Its most liked features are the IP68 water and dust resistance, fast charging, and wireless charging support. A user says: 'The camera is incredible, and the battery easily lasts a full day even with heavy use. I definitely recommend it.' This is a smartphone.""",

                "phase2_logical_reasoning.txt": """🔹 PHASE 2 — Logical Reasoning and Data Consistency Analysis

Objective: Measure the model's ability to detect missing or inconsistent fields in a record and propose web-research tasks to resolve them.

Prompt:
Act as a \"Data Integrity Auditor\". Analyze the product record below.

1) Identify fields that are null or missing.
2) By comparing registration_date and current_date, deduce which fields may be outdated.
3) Return a JSON array with web research tasks. Each task must include task_type ('MISSING_INFO' or 'UPDATE') and research_query.

Database Record:
{
    \"urun_id\": \"laptop_aurora_2023\",
    \"urun_adi\": \"AuroraBook Air 13\",
    \"marka\": \"TechNova\",
    \"kayit_tarihi\": \"2023-04-10\",
    \"guncel_tarih\": \"2025-10-22\",
    \"fiyat_tl\": 28999,
    \"teknik_ozellikler\": {
        \"islemci\": \"Intel Core i5-1230U\",
        \"ram_gb\": 8,
        \"depolama_gb\": 512,
        \"ekran_karti\": null,
        \"agirlik_kg\": 1.3
    },
    \"rakip_modeller\": []
}""",

                "phase3_analysis_synthesis.txt": """🔹 PHASE 3 — Analysis, Synthesis and Comparative Evaluation

Objective: Measure the model's ability to perform multi-criteria evaluation and scoring according to user priorities.

Prompt:
Act as a \"Product Analyst\". Using the user profile below, evaluate two smartphones.

User Profile: \"A professional who travels frequently. Priorities: long battery life (40%), high-quality camera (35%), lightweight design (25%).\"

Tasks:
1) Analyze each phone against the profile.
2) List 3 pros and 3 cons for each.
3) Compute a Suitability Score out of 10 using the weightings.
4) State which phone is more suitable and why.

Product Data:
[
    {
        \"product_name\": \"Pixel Pro 9\",
        \"battery_life_hours\": 22,
        \"camera_score_dxo\": 155,
        \"weight_grams\": 185,
        \"features\": [\"AI-assisted camera\", \"Instant translation\", \"Clean Android interface\"]
    },
    {
        \"product_name\": \"ZenPhone Max\",
        \"battery_life_hours\": 30,
        \"camera_score_dxo\": 140,
        \"weight_grams\": 220,
        \"features\": [\"Massive battery\", \"Gaming performance mode\", \"Durable casing\"]
    }
]""",

                "phase4_dialogue_recommendation.txt": """🔹 PHASE 4 — Dialogue and In-Context Recommendation (RAG)

Objective: Measure the model's ability to understand user intent and recommend the best option using only provided context.

Prompt:
Act as a \"Smart Sales Advisor Chatbot\". Use only the CONTEXT below to recommend the best product for the user's needs.

User Question:
\"Hello, I have a budget of around $1500. I play games and travel frequently; long battery life is important but I dislike very heavy devices. Which would you recommend?\"

CONTEXT:
- Product: Phantom X Gamer | Price: 1450 USD | Battery: 5500 mAh (Gaming: 10 hours) | Weight: 240g | Cooling: Vapor Chamber | Screen: 144Hz
- Product: Elite Slim V2 | Price: 1499 USD | Battery: 4500 mAh (Gaming: 6 hours) | Weight: 175g | Design: Ultra slim | Screen: 90Hz
- Product: PowerHouse One | Price: 1300 USD | Battery: 6000 mAh (Gaming: 12 hours) | Weight: 230g | Cooling: Standard | Screen: 120Hz""",

                "phase5_financial_analysis.txt": """🔹 PHASE 5 — Financial Decision Analysis (Simplified)

Objective: Measure the model's ability to perform basic arithmetic scenario comparison and return results in JSON.

Prompt:
An investor has 1,000,000 TRY and wants to buy half of a 1000 m² garden. Price per m² is 2,300 TRY. Annual interest rate is 50% (monthly compound ≈ 3.45%).

The seller offers three options:
1) Lump sum: 1,150,000 TRY now.
2) Deferred single payment: 1,250,000 TRY after 6 months.
3) Six monthly installments totaling 1,200,000 TRY.

Calculate the net cash position at the end of 6 months for each scenario assuming the investor could alternatively keep the money in the bank at the given interest rate. Return JSON:
{
    \"scenarios\": {
        \"lump_sum\": {\"net_cash_6m\": \"float\"},
        \"installments\": {\"net_cash_6m\": \"float\"},
        \"deferred\": {\"net_cash_6m\": \"float\"}
    },
    \"best_option\": \"string\"
}""",

                "phase6_code_understanding.txt": """🔹 PHASE 6 — Code Understanding and Explanation

Objective: Measure the model's ability to concisely explain what a short code snippet does (in plain English).

Prompt:
Explain the functionality of the Python code below in plain language. Provide a concise functional summary only (do not explain each line).

Code:
def analiz(dizi):
        sum = sum(dizi)
        avg = sum / len(dizi)
        difference = [(x - avg) ** 2 for x in dizi]
        variance = sum(difference) / len(dizi)
        return {\"avg\": avg, \"standard_deviation\": variance ** 0.5}

print(analiz([10, 12, 8, 15, 9]))""",

                "phase7_long_context.txt": """🔹 PHASE 7 — Long-Context Understanding (4K tokens)

Objective: Measure the model's ability to extract main themes and produce a short JSON output from a medium-length text.

Prompt:
Read the following passage and perform the tasks listed at the end. Output JSON.

Tasks:
1) Identify the top 3 themes.
2) Propose one recommendation for one of the themes.
3) State the overall tone (optimistic, critical, neutral, etc.).

Text:
Global energy transition involves not only shifting from fossil fuels to renewables but also integrating storage, smart grids and efficiency technologies. Over the past five years costs in solar and wind have dropped by more than 60%. However, energy storage remains a bottleneck. Lithium-ion production faces sustainability criticism due to environmental costs and supply chain issues. Researchers are exploring solid-state batteries and hydrogen storage. Policy inconsistencies and infrastructure gaps in some countries delay progress. The International Energy Agency recommends tripling investment by 2030.""",

                "phase8_turkish_fluency.txt": """🔹 PHASE 8 — English Fluency and Style Test

Objective: Measure the model's ability to produce a professional promotional text in English with fluent style and consistent tone.

Prompt:
Create a 3-paragraph professional feature for a technology magazine on: "Invisible but impactful uses of AI in everyday life." Use formal but fluent language, logical transitions and a corporate tone."""
        }
}

def write_test_files(selected_language='tr'):
    """Write test files based on selected language."""
    if selected_language not in TEST_FILES_CONTENT:
        cprint(f"[red]Invalid language selection: {selected_language}[/red]")
        return

    language_content = TEST_FILES_CONTENT[selected_language]

    for fname, content in language_content.items():
        p = TEST_DIR / fname
        if p.exists():
            if p.read_text().strip() == content.strip():
                cprint(f"[green]{fname} is already up to date.[/green]")
            else:
                backup_if_exists(p)
                p.write_text(content)
                cprint(f"[green]{fname} updated.[/green]")
        else:
            p.write_text(content)
            cprint(f"[green]{fname} created.[/green]")

def choose_language_interactive():
    """Interactive language selection."""
    cprint("[bold]Select test language:[/bold]")
    cprint(" 1) Turkish (Türkçe)")
    cprint(" 2) English")
    cprint(" 3) Both (Her ikisi)")

    while True:
        sel = input("Your choice (1/2/3): ").strip()
        if sel == "1":
            return 'tr'
        elif sel == "2":
            return 'en'
        elif sel == "3":
            return 'both'
        else:
            cprint("[yellow]Invalid selection. Please enter 1, 2, or 3.[/yellow]")

# ---------- venv & deps ----------
def create_venv_and_install():
    # create venv if not exists
    if not (VENV_DIR / "bin" / "python").exists():
        python_bin = shutil.which("python3") or shutil.which("python")
        if not python_bin:
            cprint("[red]python3 not found. Please install it and try again.[/red]")
            sys.exit(1)
        cprint("[cyan]Creating virtual environment...[/cyan]")
        run_cmd(f"{python_bin} -m venv {VENV_DIR}", capture=True)
    pip = VENV_DIR / "bin" / "pip"
    cprint("[cyan]Installing dependencies (in venv)...[/cyan]")
    run_cmd(f"{pip} install --upgrade pip", timeout=300)
    for pkg in PY_DEPS:
        cprint(f"[cyan]pip install {pkg}[/cyan]")
        run_cmd(f"{pip} install {pkg}", timeout=300)
    return str(VENV_DIR / "bin" / "python")

# -------- device & service helpers ----------
def detect_gpus():
    """Detect NVIDIA GPUs using nvidia-smi."""
    out, err, rc = run_cmd("nvidia-smi --query-gpu=index,name --format=csv,noheader", capture=True)
    gpus=[]
    if out:
        for line in out.splitlines():
            parts = line.split(",",1)
            if len(parts)==2:
                idx = parts[0].strip(); name = parts[1].strip()
                try:
                    gpus.append({"index": int(idx), "name": name})
                except:
                    pass
    return gpus

def list_ollama_services() -> List[str]:
    """List all ollama-worker services."""
    out, err, rc = run_cmd("systemctl list-unit-files --type=service --no-legend | grep ollama-worker || true")
    services = []
    if out:
        for line in out.splitlines():
            m = re.search(r"^(ollama-worker[-\w]+)\.service", line)
            if m:
                services.append(m.group(1))
    # also check /etc/systemd/system for files
    for p in Path("/etc/systemd/system").glob("ollama-worker*.service"):
        name = p.name.replace(".service","")
        if name not in services:
            services.append(name)
    return services

def parse_service_file(service_name: str):
    """Read ExecStart, Environment lines for port and CUDA_VISIBLE_DEVICES."""
    path_candidates = [Path("/etc/systemd/system") / (service_name + ".service"),
                       Path("/lib/systemd/system") / (service_name + ".service")]
    data = {"service": service_name, "exec": None, "env": {}, "path": None}
    for p in path_candidates:
        if p.exists():
            data["path"] = str(p)
            txt = p.read_text()
            # Get working directory if specified
            work_dir = None
            for line in txt.splitlines():
                if line.strip().startswith("WorkingDirectory="):
                    work_dir = line.split("=",1)[1].strip()
                    break
            data["work_dir"] = work_dir
            # ExecStart
            ex = None
            for line in txt.splitlines():
                if line.strip().startswith("ExecStart="):
                    ex = line.split("=",1)[1].strip()
                    break
            data["exec"] = ex
            # Environments
            envs = {}
            for line in txt.splitlines():
                if line.strip().startswith("Environment="):
                    part = line.split("=",1)[1].strip().strip('"').strip("'")
                    # Environment may be like KEY=VALUE or multiple
                    # split by space but keep = inside
                    for token in re.findall(r'(\w+=[^"\']+)', part):
                        if "=" in token:
                            k,v = token.split("=",1)
                            envs[k]=v
            data["env"] = envs
            break
    return data

# ---------- ollama list parser ----------
def parse_ollama_list_with_host(host: str) -> tuple:
    """Parse 'ollama list' output and return models with metadata."""
    try:
        out, err, rc = ollama_exec("ollama list", host, capture=True, timeout=30)
        if rc != 0 or not out:
            return [], out, err

        models = []
        lines = out.strip().split('\n')[1:]  # Skip header

        for line in lines:
            if not line.strip():
                continue
            # The `ollama list` output columns are NAME, ID, SIZE, MODIFIED - SIZE may contain spaces if units
            # We'll use a regex to capture the NAME (no spaces), ID (hex), SIZE (e.g., '3.8 GB' or '274 MB')
            m = re.match(r"^(\S+)\s+(\S+)\s+([0-9.]+\s*(?:GB|MB|TB|B))\s+", line)
            if not m:
                # fallback: split and try to find size token like 'GB' or 'MB'
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    model_id = parts[1]
                    # find token with GB/MB
                    size_token = None
                    for p in parts[2:5]:
                        if 'GB' in p or 'MB' in p or 'TB' in p or 'B' == p:
                            size_token = p
                            break
                    size_str = size_token or ''
                else:
                    continue
            else:
                name = m.group(1)
                model_id = m.group(2)
                size_str = m.group(3)

            # Normalize size string and parse to MB
            size_mb = None
            try:
                s = size_str.strip()
                if s.endswith('GB'):
                    size_mb = float(s.replace('GB', '').strip()) * 1024
                elif s.endswith('MB'):
                    size_mb = float(s.replace('MB', '').strip())
                elif s.endswith('TB'):
                    size_mb = float(s.replace('TB', '').strip()) * 1024 * 1024
                elif s.endswith('B'):
                    size_mb = float(s.replace('B', '').strip()) / (1024 * 1024)
            except Exception:
                size_mb = None

            models.append({
                'name': name,
                'id': model_id,
                'size_mb': size_mb,
                'size_str': size_str
            })

        return models, out, err
    except Exception as e:
        return [], "", str(e)

def enrich_models_with_sizes(models: List[dict]) -> List[dict]:
    """Enrich models with file sizes if not available."""
    enriched = []
    for model in models:
        if model.get('size_mb') is None:
            # Try to find file size
            size_mb = find_model_file_size(model['name'])
            if size_mb:
                model['size_mb'] = size_mb
        enriched.append(model)
    return enriched

# ---------- model size helpers ----------
# Model dosyalarının gerçek konumu servis konfigürasyonundan alınıyor
MODEL_DIRS = [
    "/data/ollama/models/blobs",  # Ana model dizini
    "/data/ollama/.ollama/models/blobs",  # Alternatif konum
    "/usr/share/ollama/.ollama/models/blobs",
    "/var/lib/ollama/models/blobs",
    "~/.ollama/models/blobs"
]

def find_model_file_size(model_name: str) -> float:
    """Find model file size in MB."""
    for model_dir in MODEL_DIRS:
        try:
            model_path = Path(model_dir)
            if model_path.exists():
                # Look for model files (usually sha256:...)
                for blob_file in model_path.glob("sha256-*"):
                    try:
                        size_mb = blob_file.stat().st_size / (1024 * 1024)
                        return size_mb
                    except:
                        continue
        except:
            continue
    return None

def estimate_load_time_mb(size_mb: float) -> int:
    """Estimate model load time in seconds based on size."""
    if not size_mb:
        return 30  # default

    # Rough estimation: ~10 seconds per GB + base time
    base_time = 10
    per_gb_time = 8
    estimated = base_time + (size_mb / 1024) * per_gb_time
    return max(15, int(estimated))

# ---------- logging helpers ----------
def write_benchmark_log(message: str, log_file_path: str = None):
    """Write message to benchmark log file with dynamic naming."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Eğer log_file_path belirtilmemişse dinamik isim oluştur
        if log_file_path is None:
            timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
            log_file_path = LOGS_DIR / f"log_{timestamp}.txt"

        # Log dosyası path'ini Path objesi olarak al
        log_file = Path(log_file_path)

        # Timestamp formatını log mesajı için de tutarlı hale getir
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        cprint(f"[red]Log Writing Error: {e}[/red]")

def create_benchmark_log_file():
    """Create a new benchmark log file with timestamp and return its path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    log_file_path = LOGS_DIR / f"log_{timestamp}.txt"

    # Log dosyasının başlangıcını kaydet
    write_benchmark_log("=== Benchmark test started ===", str(log_file_path))
    write_benchmark_log(f"Log file: {log_file_path}", str(log_file_path))

    return str(log_file_path)

# ---------- telemetry worker (enhanced) ----------
def telemetry_worker(stop_event, telemetry_file: Path):
    """Enhanced GPU telemetry worker with comprehensive metrics."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        # Enhanced fieldnames for comprehensive telemetry
        fieldnames = [
            'timestamp', 'gpu_index', 'util_gpu_pct', 'util_memory_pct',
            'mem_used_MiB', 'mem_total_MiB', 'mem_free_MiB',
            'temp_C', 'power_W', 'power_limit_W',
            'clock_core_MHz', 'clock_memory_MHz', 'clock_sm_MHz',
            'fan_speed_pct', 'pcie_tx_MBps', 'pcie_rx_MBps',
            'encoder_util_pct', 'decoder_util_pct'
        ]

        with open(telemetry_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            write_benchmark_log(f"Telemetry worker (nvidia-ml-py) started. Output to {telemetry_file}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))

            while not stop_event.is_set():
                try:
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                        # Basic metrics
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                        # Power metrics
                        try:
                            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                        except:
                            power = 0

                        try:
                            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                        except:
                            power_limit = 0

                        # Clock speeds
                        try:
                            clock_core = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                        except:
                            clock_core = 0

                        try:
                            clock_memory = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                        except:
                            clock_memory = 0

                        try:
                            clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        except:
                            clock_sm = 0

                        # Fan speed
                        try:
                            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                        except:
                            fan_speed = 0

                        # PCIe throughput (if available)
                        try:
                            pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / (1024*1024)  # MB/s
                            pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / (1024*1024)  # MB/s
                        except:
                            pcie_tx = 0
                            pcie_rx = 0

                        # Encoder/Decoder utilization (if available)
                        try:
                            encoder_util = pynvml.nvmlDeviceGetEncoderUtilization(handle)[0]
                            decoder_util = pynvml.nvmlDeviceGetDecoderUtilization(handle)[0]
                        except:
                            encoder_util = 0
                            decoder_util = 0

                        writer.writerow({
                            'timestamp': datetime.now().isoformat(),
                            'gpu_index': i,
                            'util_gpu_pct': util.gpu,
                            'util_memory_pct': util.memory,
                            'mem_used_MiB': mem.used // (1024 * 1024),
                            'mem_total_MiB': mem.total // (1024 * 1024),
                            'mem_free_MiB': (mem.total - mem.used) // (1024 * 1024),
                            'temp_C': temp,
                            'power_W': power,
                            'power_limit_W': power_limit,
                            'clock_core_MHz': clock_core,
                            'clock_memory_MHz': clock_memory,
                            'clock_sm_MHz': clock_sm,
                            'fan_speed_pct': fan_speed,
                            'pcie_tx_MBps': pcie_tx,
                            'pcie_rx_MBps': pcie_rx,
                            'encoder_util_pct': encoder_util,
                            'decoder_util_pct': decoder_util
                        })
                except Exception as e_inner:
                    write_benchmark_log(f"pynvml data collection error (GPU {i}): {e_inner}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))
                    pass  # Continue on errors

                time.sleep(GPU_LOG_INTERVAL)

    except Exception as e_pynvml_init:
        cprint(f"[yellow]pynvml cant start, selected nvidia-smi: {e_pynvml_init}[/yellow]")
        write_benchmark_log(f"pynvml cant start, selected nvidia-smi: {e_pynvml_init}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))
        # Enhanced fallback to nvidia-smi with more metrics
        try:
            # Write header for nvidia-smi fallback
            fallback_fieldnames = [
                'timestamp', 'gpu_index', 'utilization.gpu', 'utilization.memory',
                'memory.used', 'memory.total', 'temperature.gpu', 'power.draw',
                'clocks.current.graphics', 'clocks.current.memory', 'clocks.current.sm',
                'fan.speed', 'pcie.link.gen.current', 'pcie.link.width.current',
                'encoder.stats.sessionCount', 'encoder.stats.averageFps',
                'decoder.stats.sessionCount', 'decoder.stats.averageFps'
            ]
            with open(telemetry_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fallback_fieldnames)
            write_benchmark_log(f"Telemetry worker (nvidia-smi fallback) started. Output to {telemetry_file}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))

            while not stop_event.is_set():
                try:
                    # Enhanced nvidia-smi query with more metrics
                    cmd = (
                        "nvidia-smi --query-gpu="
                        "timestamp,"
                        "index,"
                        "utilization.gpu,"
                        "utilization.memory,"
                        "memory.used,"
                        "memory.total,"
                        "temperature.gpu,"
                        "power.draw,"
                        "clocks.current.graphics,"
                        "clocks.current.memory,"
                        "clocks.current.sm,"
                        "fan.speed,"
                        "pcie.link.gen.current,"
                        "pcie.link.width.current,"
                        "encoder.stats.sessionCount,"
                        "encoder.stats.averageFps,"
                        "decoder.stats.sessionCount,"
                        "decoder.stats.averageFps"
                        " --format=csv,noheader,nounits"
                    )
                    out, err, rc = run_cmd(cmd, capture=True)
                    write_benchmark_log(f"nvidia-smi command: {cmd}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))
                    write_benchmark_log(f"nvidia-smi stdout: {out}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))
                    write_benchmark_log(f"nvidia-smi stderr: {err}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))
                    write_benchmark_log(f"nvidia-smi rc: {rc}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))


                    if rc == 0 and out:
                        lines = out.strip().split('\n')
                        with open(telemetry_file, 'a', newline='') as csvfile:
                            for i, line in enumerate(lines):
                                if line.strip():
                                    parts = line.split(',')
                                    if len(parts) >= 18:
                                        # Parse available metrics, fill missing with 0
                                        timestamp = parts[0] + ' ' + parts[1] if len(parts) > 1 else datetime.now().isoformat()
                                        gpu_index = parts[2] if len(parts) > 2 else str(i)
                                        util_gpu = parts[3] if len(parts) > 3 else '0'
                                        util_mem = parts[4] if len(parts) > 4 else '0'
                                        mem_used = parts[5] if len(parts) > 5 else '0'
                                        mem_total = parts[6] if len(parts) > 6 else '0'
                                        temp = parts[7] if len(parts) > 7 else '0'
                                        power = parts[8] if len(parts) > 8 else '0'
                                        clock_core = parts[9] if len(parts) > 9 else '0'
                                        clock_mem = parts[10] if len(parts) > 10 else '0'
                                        clock_sm = parts[11] if len(parts) > 11 else '0'
                                        fan_speed = parts[12] if len(parts) > 12 else '0'
                                        pcie_gen = parts[13] if len(parts) > 13 else '0'
                                        pcie_width = parts[14] if len(parts) > 14 else '0'
                                        encoder_sessions = parts[15] if len(parts) > 15 else '0'
                                        encoder_fps = parts[16] if len(parts) > 16 else '0'
                                        decoder_sessions = parts[17] if len(parts) > 17 else '0'
                                        decoder_fps = parts[18] if len(parts) > 18 else '0'

                                        csvfile.write(f"{datetime.now().isoformat()},{gpu_index},{util_gpu},{util_mem},{mem_used},{mem_total},{temp},{power},{clock_core},{clock_mem},{clock_sm},{fan_speed},{pcie_gen},{pcie_width},{encoder_sessions},{encoder_fps},{decoder_sessions},{decoder_fps}\n")
                    else:
                        write_benchmark_log(f"nvidia-smi output empty or error rc: {rc}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))
                except Exception as e_inner_fallback:
                    write_benchmark_log(f"nvidia-smi data collection error: {e_inner_fallback}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))
                    pass
                time.sleep(GPU_LOG_INTERVAL)
        except Exception as e2_fallback_init:
            cprint(f"[red]Enhanced telemetry worker cant startup: {e2_fallback_init}[/red]")
            write_benchmark_log(f"Enhanced telemetry worker cant startup: {e2_fallback_init}", log_file_path=str(LOGS_DIR / "telemetry_debug.log"))
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

# ---------- ollama command wrapper ----------
def ollama_exec(cmd: str, host: str, user: str = "ollama", capture=True, timeout=None):
    """Enhanced ollama command execution with service-specific environment."""
    # Build environment
    env = {
        "OLLAMA_HOST": host,
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/var/lib/ollama",
        "OLLAMA_MODELS": "/data/ollama/models"  # Servis configinden alınan
    }
    
    # For run commands, ensure using full binary path
    if "ollama run" in cmd and not cmd.startswith("/usr/local/bin/"):
        cmd = cmd.replace("ollama run", "/usr/local/bin/ollama run")
    
    # Build sudo command with environment preservation
    env_str = " ".join(f'{k}="{v}"' for k, v in env.items())
    safe_cmd = f"sudo -E -u {user} env {env_str} {cmd}"
    
    write_benchmark_log(f"Executing: {safe_cmd}")
    return run_cmd(safe_cmd, timeout=timeout, capture=capture)

def run_tests_for_models(models: List[str], host: str, log_file_path: str = None):
    # ensure results CSV exists - dinamik isimlendirme ile
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Test başlangıç zamanını al (CSV ve telemetry için aynı timestamp kullan)
    start_all = datetime.now()
    ts = start_all.strftime("%d%m%Y_%H%M%S")

    # CSV dosyasını dinamik isimle oluştur
    results_csv_path = RESULTS_DIR / f"benchmark_results_{ts}.csv"

    # Eğer dosya yoksa başlık satırı ile oluştur
    if not results_csv_path.exists():
        import pandas as pd
        pd.DataFrame(columns=["timestamp","model","test_case","duration_s","output_chars","output"]).to_csv(results_csv_path, index=False)

    # telemetry start - aynı timestamp ile
    telemetry_file = TELEMETRY_DIR / f"gpu_usage_{ts}.csv"
    stop_event = threading.Event()
    t_thread = threading.Thread(target=telemetry_worker, args=(stop_event, telemetry_file), daemon=True)
    t_thread.start()

    # Log dosyası path'ini kullan veya oluştur
    if log_file_path is None:
        log_file_path = create_benchmark_log_file()

    write_benchmark_log(f"Benchmark started host={host} models={','.join(models)}", log_file_path)
    write_benchmark_log(f"Results CSV: {results_csv_path}", log_file_path)
    write_benchmark_log(f"Telemetry CSV: {telemetry_file}", log_file_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        for mi, model in enumerate(models, start=1):
            model_task = progress.add_task(f"[cyan]Model {mi}/{len(models)}: {model}", total=100)
            progress.update(model_task, advance=0)

            # Model yükleme aşaması
            size_mb = None
            if MODEL_META.exists():
                try:
                    d=json.loads(MODEL_META.read_text())
                    size_mb = d.get(model)
                except:
                    pass
            if size_mb is None:
                size_mb = find_model_file_size(model)
            
            est = estimate_load_time_mb(size_mb)
            progress.update(model_task, description=f"[yellow]Loading model: {model} ({size_mb if size_mb else 'unknown'} MB)")
            
            # Enhanced ping and verification
            ping_ok = False
            for attempt in range(PING_RETRIES):
                progress.update(model_task, description=f"[yellow]Model ping attempt {attempt+1}: {model}")
                
                # Try to pull model first if needed
                pull_cmd = f"/usr/local/bin/ollama pull {model}"
                out, err, rc = ollama_exec(pull_cmd, host, capture=True, timeout=300)
                if rc != 0:
                    progress.update(model_task, description=f"[red]Model pull error: {err}")
                    time.sleep(PING_RETRY_WAIT)
                    continue
                
                # Then try ping with a simple prompt
                ping_prompt = "Hello"
                out, err, rc = ollama_exec(f"/usr/local/bin/ollama run {model} {ping_prompt}", host, capture=True, timeout=est+30)
                if rc == 0 and out and out.strip():
                    ping_ok = True
                    progress.update(model_task, advance=20)
                    break
                else:
                    progress.update(model_task, description=f"[yellow]Model loading failed: {err}")
                time.sleep(PING_RETRY_WAIT)
            
            if not ping_ok:
                progress.update(model_task, description=f"[red]Model {model} could not be loaded!")
                write_benchmark_log(f"Model {model} ping failed")
                continue

            # Test files
            progress.update(model_task, description=f"[green]Model ready: {model} - Starting tests")
            test_files = sorted(TEST_DIR.glob("*.txt"))
            if not test_files:
                progress.update(model_task, description=f"[red]Error: No test files found. {TEST_DIR} is empty.")
                write_benchmark_log(f"No test files found in {TEST_DIR}. Aborting tests for model {model}.", log_file_path)
                # advance to mark as failed and continue to next model
                progress.update(model_task, advance=100)
                continue
            test_progress = 60 / len(test_files)  # Remaining 60% divided by tests
            
            for tfile in test_files:
                progress.update(model_task, description=f"[blue]Running test: {tfile.name}")
                prompt = tfile.read_text()
                start_time = datetime.now()

                try:
                    # Use subprocess with stdin to avoid shell quoting issues and ensure prompt is delivered reliably.
                    cmd_list = [
                        'sudo', '-E', '-u', 'ollama',
                        'env', f'OLLAMA_HOST={host}',
                        '/usr/local/bin/ollama', 'run', model
                    ]
                    p = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, input=prompt, timeout=MAX_RESPONSE_TIME)
                    out, err = p.stdout, p.stderr
                    rc = p.returncode

                    duration = (datetime.now() - start_time).total_seconds()

                    # Save results to CSV - dinamik dosya ismi kullan
                    import pandas as pd
                    timestamp = start_time.isoformat()

                    # Create result entry
                    result_data = {
                        'timestamp': timestamp,
                        'model': model,
                        'test_case': tfile.name,
                        'duration_s': duration,
                        'output_chars': len(out) if out else 0,
                        'output': out[:MAX_OUTPUT_CHARS] if out else ""
                    }

                    # Append to CSV - dinamik dosya kullan
                    df = pd.DataFrame([result_data])
                    if results_csv_path.exists():
                        df.to_csv(results_csv_path, mode='a', header=False, index=False)
                    else:
                        df.to_csv(results_csv_path, index=False)

                    progress.update(model_task, advance=test_progress)
                except subprocess.TimeoutExpired:
                    duration = (datetime.now() - start_time).total_seconds()
                    progress.update(model_task, description=f"[red]Test timeout: {tfile.name}")

                    # Save timeout result - dinamik dosya kullan
                    import pandas as pd
                    timestamp = start_time.isoformat()

                    result_data = {
                        'timestamp': timestamp,
                        'model': model,
                        'test_case': tfile.name,
                        'duration_s': duration,
                        'output_chars': 0,
                        'output': "TIMEOUT"
                    }

                    df = pd.DataFrame([result_data])
                    if results_csv_path.exists():
                        df.to_csv(results_csv_path, mode='a', header=False, index=False)
                    else:
                        df.to_csv(results_csv_path, index=False)

                time.sleep(TEST_WAIT)

            # Stop model
                progress.update(model_task, description="[yellow]Stopping model...")
            out,err,rc = ollama_exec(f"ollama stop {model}", host, capture=True)
            
            # Verify stop
            stopped = False
            for _ in range(6):
                time.sleep(3)
                ps_out,_,_ = ollama_exec("ollama ps", host, capture=True)
                if model not in (ps_out or ""):
                    stopped = True
                    break
            
            progress.update(model_task, advance=20)
            if stopped:
                    progress.update(model_task, description=f"[green]Model {model} completed!")
            
            # Cooldown
            cooldown_task = progress.add_task(f"[cyan]Cooldown period", total=COOLDOWN_AFTER_MODEL)
            for _ in range(COOLDOWN_AFTER_MODEL):
                time.sleep(1)
                progress.update(cooldown_task, advance=1)
            progress.remove_task(cooldown_task)

    # finalize
    stop_event.set()
    t_thread.join(timeout=5)
    total_time = datetime.now() - start_all
    write_benchmark_log(f"Benchmark completed total_seconds={total_time.total_seconds()}")
    cprint(f"[bold green]Benchmark completed. Total duration: {str(total_time)}[/bold green]")
    # generate master dashboard (consolidated)
    try:
        html_report_path = RESULTS_DIR / f"benchmark_master_dashboard_{ts}.html"
        # Generate a single master dashboard that scans the results and telemetry folders
        generate_master_dashboard(RESULTS_DIR, TELEMETRY_DIR, html_report_path, log_file_path)
        cprint(f"[bold]Master dashboard generated: {html_report_path}[/bold]")
        write_benchmark_log(f"Master dashboard generated: {html_report_path}", log_file_path)
    except Exception as e:
        cprint(f"[red]Failed to generate master dashboard: {e}[/red]")
        write_benchmark_log(f"Failed to generate master dashboard: {e}", log_file_path)

# Legacy advanced and comparative dashboard generators removed.
# The codebase now uses `generate_master_dashboard` (already present) for consolidated reporting.
# If you need any of the removed per-test or comparative sections restored, I can reintroduce
# smaller, testable helpers on request.


def generate_master_dashboard(results_dir: Path, telemetry_dir: Path, out_html_path: Path, log_file_path: str = None):
    """Compact master dashboard generator: combines benchmark CSVs and matching telemetry files.

    This produces a small interactive HTML that lets you pick a test run and model, view
    response durations and overlay telemetry (util/temp/power) when available.
    """
    try:
        import pandas as pd
        import json
    except Exception as e:
        raise Exception(f"Master dashboard requires pandas: {e}")

    results_dir = Path(results_dir)
    telemetry_dir = Path(telemetry_dir)

    files = sorted(results_dir.glob('benchmark_results_*.csv'))
    if not files:
        raise Exception(f"No benchmark_results_*.csv files found in {results_dir}")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['run_file'] = f.name
            # try to ensure timestamp column is parseable
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise Exception("No readable result CSVs found")

    all_res = pd.concat(frames, ignore_index=True)

    # Build telemetry map: try to find telemetry file with matching timestamp fragment
    telemetry_map = {}
    for f in files:
        ts_frag = f.stem.replace('benchmark_results_', '')
        tfile = telemetry_dir / f"gpu_usage_{ts_frag}.csv"
        if tfile.exists():
            try:
                tdf = pd.read_csv(tfile)
                if 'timestamp' in tdf.columns:
                    tdf['timestamp'] = pd.to_datetime(tdf['timestamp'], errors='coerce').astype(str)
                telemetry_map[f.name] = tdf.to_dict(orient='records')
            except Exception:
                telemetry_map[f.name] = None
        else:
            telemetry_map[f.name] = None

    # Aggregate results per run_file, test_case, model
    agg = all_res.groupby(['run_file', 'test_case', 'model']).agg(
        start_time=('timestamp', 'min'),
        duration_s=('duration_s', 'mean'),
        count=('timestamp', 'count')
    ).reset_index()

    # Convert timestamp to string to avoid JSON serialization issues
    agg['start_time'] = agg['start_time'].astype(str)

    # Build a structured payload grouped by run -> test_case -> models
    payload = {'runs': {}, 'telemetry': {}}

    for run_file in sorted(agg['run_file'].unique()):
        run_df = agg[agg['run_file'] == run_file]
        tests = {}
        for test_case in sorted(run_df['test_case'].unique()):
            r = run_df[run_df['test_case'] == test_case]
            models_list = []
            for _, row in r.iterrows():
                models_list.append({
                    'model': row['model'],
                    'duration_s': float(row['duration_s']) if not pd.isna(row['duration_s']) else None,
                    'start_time': row['start_time'],
                    'count': int(row['count'])
                })
            tests[test_case] = models_list
        payload['runs'][run_file] = tests

    # Attach telemetry aggregates per run_file -> model -> test_case if telemetry exists
    for f in files:
        ts_frag = f.stem.replace('benchmark_results_', '')
        tfile = telemetry_dir / f"gpu_usage_{ts_frag}.csv"
        if tfile.exists():
            try:
                tdf = pd.read_csv(tfile)
                # Normalize telemetry column names
                col_map = {c.lower(): c for c in tdf.columns}
                # Coerce numeric columns
                for c in tdf.columns:
                    try:
                        tdf[c] = pd.to_numeric(tdf[c], errors='coerce')
                    except:
                        pass

                # Simple telemetry aggregations by model not available: aggregate overall stats per run
                telemetry_agg = {}
                if 'timestamp' in tdf.columns:
                    tdf['timestamp'] = pd.to_datetime(tdf['timestamp'], errors='coerce')

                # Compute overall averages/max for key columns if present
                # try common names
                def first_col_like(df, candidates):
                    for cand in candidates:
                        for col in df.columns:
                            if cand in col.lower():
                                return col
                    return None

                util_col = first_col_like(tdf, ['util', 'utilization', 'util_gpu', 'utilization.gpu'])
                temp_col = first_col_like(tdf, ['temp', 'temperature'])
                power_col = first_col_like(tdf, ['power', 'power.draw', 'power_w'])

                if util_col:
                    telemetry_agg['avg_util'] = float(tdf[util_col].mean(skipna=True))
                if temp_col:
                    telemetry_agg['max_temp'] = float(tdf[temp_col].max(skipna=True))
                if power_col:
                    telemetry_agg['avg_power'] = float(tdf[power_col].mean(skipna=True))

                payload['telemetry'][f.name] = telemetry_agg
            except Exception:
                payload['telemetry'][f.name] = None
        else:
            payload['telemetry'][f.name] = None

    # Copy run-level telemetry aggregates into each model entry for convenience
    for run_file, tests in payload['runs'].items():
        run_tel = payload['telemetry'].get(run_file)
        for test_case, models_list in tests.items():
            for m in models_list:
                # attach telemetry aggregates (run-level) if model-level not available
                if run_tel:
                    m['telemetry'] = run_tel
                else:
                    m['telemetry'] = None

        html = [
                '<!doctype html>',
                '<html><head><meta charset="utf-8"><title>Master Benchmark Dashboard</title>',
                '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
                '<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px}</style>',
                '</head><body>'
        ]

        html.append('<h1>Master Benchmark Dashboard</h1>')
        html.append('<div><label>Test Run: <select id="run"></select></label> <label>Model/Test: <select id="model"><option value="ALL">ALL</option></select></label></div>')
        html.append('<div id="summary"></div>')
        html.append('<div id="resp" style="height:420px"></div>')
        html.append('<div id="util" style="height:180px"></div>')

        # embed payload
        html.append(f"<script>const payload = {json.dumps(payload)};</script>")

        # client script: render grouped bars for durations and telemetry overlays
        client = r'''
        <script>
        function init(){
            const runs = Object.keys(payload.runs || {});
            const runSel = document.getElementById('run');
            runs.forEach(r=>{ const o=document.createElement('option'); o.value=r; o.text=r; runSel.appendChild(o); });
            runSel.addEventListener('change', onRunChange);
            const modelSel = document.getElementById('model');
            modelSel.addEventListener('change', render);
            if(runs.length) onRunChange();
        }

        function onRunChange(){
            const selRun = document.getElementById('run').value;
            const tests = Object.keys((payload.runs && payload.runs[selRun]) || {});
            const testSel = document.getElementById('model');
            testSel.innerHTML = '<option value="ALL">ALL</option>';
            tests.forEach(t=>{ const o=document.createElement('option'); o.value=t; o.text=t; testSel.appendChild(o); });
            render();
        }

        function render(){
            const selRun = document.getElementById('run').value;
            const selTest = document.getElementById('model').value;
            const runObj = (payload.runs && payload.runs[selRun]) || {};
            const tests = selTest === 'ALL' ? Object.keys(runObj) : [selTest];

            // Build grouped arrays per test_case
            let names = [];
            let durations = [];
            let avgUtils = [];
            let avgPows = [];

            tests.forEach(test_case => {
                const mlist = runObj[test_case] || [];
                mlist.forEach(m => {
                    names.push(m.model + ' / ' + test_case);
                    durations.push(m.duration_s || 0);
                    avgUtils.push(m.telemetry && m.telemetry.avg_util ? m.telemetry.avg_util : null);
                    avgPows.push(m.telemetry && m.telemetry.avg_power ? m.telemetry.avg_power : null);
                });
            });

            const traces = [];
            traces.push({ x: names, y: durations, type: 'bar', name: 'Duration (s)', marker: {color:'#1f77b4'} });

            if(avgUtils.some(v=>v!==null)){
                traces.push({ x: names, y: avgUtils, mode: 'lines+markers', name: 'Avg GPU Util (%)', yaxis: 'y2', marker:{color:'#ff7f0e'} });
            }
            if(avgPows.some(v=>v!==null)){
                traces.push({ x: names, y: avgPows, mode: 'lines+markers', name: 'Avg Power (W)', yaxis: 'y2', marker:{color:'#2ca02c'} });
            }

            const layout = {
                title: 'Response Duration (s) and Telemetry Overlays',
                yaxis: { title: 'Duration (s)' },
                yaxis2: { title: 'Telemetry (util/power)', overlaying: 'y', side: 'right' },
                margin: { t: 40, b: 140 }
            };

            Plotly.newPlot('resp', traces, layout, {responsive: true});

            // telemetry summary
            const telDiv = document.getElementById('util');
            const telSummary = payload.telemetry && payload.telemetry[selRun];
            if(telSummary){
                telDiv.innerHTML = `<div style="padding:8px;color:#333"><b>Telemetry summary for run:</b> Avg Util: ${telSummary.avg_util||'N/A'} | Max Temp: ${telSummary.max_temp||'N/A'} | Avg Power: ${telSummary.avg_power||'N/A'}</div>`;
            } else {
                telDiv.innerHTML = '<div style="padding:8px;color:#666">No telemetry aggregates for this run.</div>';
            }
        }
        window.addEventListener('load', init);
        </script>
        '''

        html.append(client)
        html.append('</body></html>')

        out_html_path.write_text('\n'.join(html))
        return str(out_html_path)


# -------- interactive selection helpers ----------
def choose_service_interactive(services: List[str]):
    cprint("[bold]Detected Ollama worker services:[/bold]")
    service_infos = []
    for i, s in enumerate(services, start=1):
        info = parse_service_file(s)
        port = None
        if info["env"].get("OLLAMA_HOST"):
            m = re.search(r":(\d+)", info["env"].get("OLLAMA_HOST"))
            if m:
                port = m.group(1)
        service_infos.append({"name": s, "port": port, "env": info["env"], "exec": info["exec"], "path": info["path"]})
        cprint(f" {i}) {s}  port={port}  exec={info['exec']}")
    cprint(" m) Manual host (e.g., 127.0.0.1:11435)")
    cprint(" d) Default host (127.0.0.1:11434)")
    sel = input("Your choice: ").strip()
    if sel.lower() == "m":
        host = input("OLLAMA_HOST address (e.g., 127.0.0.1:11435): ").strip()
        return host
    if sel.lower() == "d" or sel == "":
        return "127.0.0.1:11434"
    try:
        idx = int(sel)-1
        chosen = service_infos[idx]
        port = chosen.get("port") or "11434"
        return f"127.0.0.1:{port}"
    except Exception:
        cprint("[red]Invalid selection. Default host will be used.[/red]")
        return "127.0.0.1:11434"

def choose_models_interactive(models: List[dict]):
    if not models:
        cprint("[red]No models found.[/red]")
        return []
    cprint("[bold]Installed models (sorted by size):[/bold]")
    for i,m in enumerate(models, start=1):
        s = f"{m['size_mb']:.1f} MB" if m['size_mb'] else "size unknown"
        cprint(f" {i}) {m['name']} ({s})")
    cprint("Options: 'A' = all, '1,3,5' or range '1-3'. ('q' = cancel)")
    sel = input("Your choice: ").strip()
    if sel.lower() == "a":
        return [m["name"] for m in models]
    if sel.lower() == "q" or sel=="":
        return []
    chosen = set()
    tokens = sel.split(",")
    for t in tokens:
        t=t.strip()
        if "-" in t:
            a,b = t.split("-",1)
            try:
                a=int(a); b=int(b)
                for i in range(a, b+1):
                    if 1<=i<=len(models):
                        chosen.add(models[i-1]["name"])
            except:
                pass
        else:
            try:
                i=int(t)
                if 1<=i<=len(models):
                    chosen.add(models[i-1]["name"])
            except:
                pass
    if not chosen:
        cprint("[yellow]Invalid selection or empty. Selecting all models.[/yellow]")
        return [m["name"] for m in models]
    return list(chosen)

def choose_gpu_mode_interactive(gpus):
    if not gpus:
        cprint("[yellow]No GPUs found. Switching to CPU-only mode.[/yellow]")
        return None
    cprint("[bold]GPUs detected:[/bold]")
    for g in gpus:
        cprint(f" {g['index']}) {g['name']}")
    cprint(" 1) CPU only")
    cprint(" 2) Use All GPUs (run on orchestrator worker-all)")
    cprint(" 3) Select specific GPUs (e.g. 0,2)")
    sel = input("Your choice (1/2/3): ").strip()
    if sel == "1":
        return None
    if sel == "2":
        return ",".join(str(g["index"]) for g in gpus)
    if sel == "3":
        s = input("Enter GPU indices (e.g., 0,2): ").strip()
        parts=[p.strip() for p in s.split(",") if p.strip().isdigit()]
        return ",".join(parts)
    cprint("[yellow]Invalid selection. Defaulting to All GPUs.[/yellow]")
    return ",".join(str(g["index"]) for g in gpus)

# ---------- main ----------
def main():
    # require root for service introspection & to spawn background runner under user
    if os.geteuid() != 0:
        cprint("[red]This script requires sudo/root privileges for initial execution.[/red]")
        cprint("sudo python3 " + str(Path(__file__).absolute()))
        sys.exit(1)

    # Check if running inside venv, if not, re-execute in venv
    venv_python = VENV_DIR / "bin" / "python"
    if venv_python.exists() and sys.executable != str(venv_python):
        cprint(f"[cyan]Restarting in virtual environment: {venv_python}[/cyan]")
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
        # os.execv will replace the current process, so no code below this will run in the current process.

    cprint("[bold]Ollama Benchmark Suite v6 — Initializing[/bold]")
    prepare_dirs()

    # Language selection menu
    selected_language = choose_language_interactive()
    cprint(f"[green]Selected language: {selected_language}[/green]")

    # Create test files for selected language
    if selected_language == 'both':
        # Create files for both languages
        write_test_files('tr')
        write_test_files('en')
        cprint("[green]Turkish and English test files created.[/green]")
    else:
        write_test_files(selected_language)
        cprint(f"[green]{selected_language} test files created.[/green]")
    # detect services
    services = list_ollama_services()
    if not services:
        cprint("[yellow]No ollama-worker services detected. You will be asked to enter host manually.[/yellow]")
    selected_host = choose_service_interactive(services) if services else input("Enter OLLAMA_HOST (e.g., 127.0.0.1:11435): ").strip()
    cprint(f"[green]Selected OLLAMA_HOST: {selected_host}[/green]")
    # detect GPUs & choose mode
    gpus = detect_gpus()
    chosen_cuda = choose_gpu_mode_interactive(gpus)
    if chosen_cuda is None:
        # CPU only
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen_cuda
    # create venv and install
    create_venv_and_install() # This will ensure deps are installed, but the script is already running in venv
    # list models from the selected host
    models, out, err = parse_ollama_list_with_host(selected_host)
    if not models:
        cprint("[yellow]Warning: No models found with 'ollama list' or service is unreachable.[/yellow]")
        if input("Do you want to continue? (y/N): ").strip().lower() != "y":
            cprint("Operation cancelled.")
            sys.exit(0)
    else:
        models = enrich_models_with_sizes(models)
    # interactive model selection
    selected_models = choose_models_interactive(models)
    if not selected_models:
        cprint("[yellow]No models selected. Exiting.[/yellow]")
        sys.exit(0)
    # save metadata
    dmeta = {m['name']: m['size_mb'] for m in models}
    MODEL_META.write_text(json.dumps(dmeta))
    SELECTED_MODELS_FILE.write_text(json.dumps(selected_models))
    cprint(f"[green]Selected models saved: {selected_models}[/green]")
    # confirmation
    if input("Start benchmark test? (Y/n): ").strip().lower() in ("y",""):
        # Set env vars
        os.environ["OLLAMA_HOST"] = selected_host
        if chosen_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = chosen_cuda
            
        # Run directly instead of background
        cprint("[cyan]Benchmark starting...[/cyan]")
        run_tests_for_models(selected_models, selected_host)
    else:
        cprint("[yellow]Benchmark not started. Selected models saved to file.[/yellow]")

if __name__ == "__main__":
    # Ensure VENV_DIR is defined before main() is called or re-executed
    # This part is crucial for the re-execution logic
    _sudo_user = os.environ.get("SUDO_USER")
    if _sudo_user:
        try:
            _user_home = Path(pwd.getpwnam(_sudo_user).pw_dir)
        except Exception:
            _user_home = Path(os.path.expanduser(f"~{_sudo_user}")) if _sudo_user else Path.home()
    else:
        _user_home = Path.home()
    _base = _user_home / "ollama_benchmarks"
    VENV_DIR = _base / "venv" # Define VENV_DIR here for re-execution check

    if "--run-host" in sys.argv:
        # background runner
        try:
            idx = sys.argv.index("--run-host")+1
            host = sys.argv[idx]
        except:
            host = "127.0.0.1:11435"
        # load selected models
        if SELECTED_MODELS_FILE.exists():
            sel = json.loads(SELECTED_MODELS_FILE.read_text())
        else:
            cprint("[red]Seçili modeller bulunamadı.[/red]")
            sys.exit(1)
        run_tests_for_models(sel, host)
    elif "--create-comparative-dashboard" in sys.argv:
        # The old comparative dashboard generator was removed to avoid duplication.
        cprint("[yellow]Karşılaştırmalı dashboard oluşturma fonksiyonu kaldırıldı. Lütfen master dashboard'ı kullanın.[/yellow]")
        sys.exit(0)
    else:
        main()
