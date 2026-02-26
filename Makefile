# litmus Makefile
#
# Targets:
#   make build        - Build Rust CLI
#   make collector    - Build Go sample collector
#   make data         - Download benign samples
#   make features     - Extract features from samples
#   make train        - Train XGBoost model
#   make evaluate     - Evaluate model performance
#   make all          - Full pipeline: data -> features -> train -> evaluate
#   make clean        - Remove build artifacts

SHELL := /bin/bash
.PHONY: all build collector data features train evaluate test-eval clean venv lint test

# Configuration
DATA_DIR ?= $(HOME)/data/known-good
MALWARE_DIR ?= $(HOME)/data/malware
TEST_DIR ?= ../test_samples_NOT_TRAINING
FEATURES_DIR ?= training/data
MODEL_DIR ?= models
VENV_DIR ?= training/.venv
PYTHON ?= $(VENV_DIR)/bin/python
MAX_PACKAGES ?= 50
WORKERS ?= 4
LIMIT ?= 0
EXTENSIONS ?=
EXTENSION ?=
CACHE_DIR ?= training/cache

# cleave binary location
CLEAVE_DIR ?= ../cleave
CLEAVE_BIN ?= $(CLEAVE_DIR)/target/release/cleave

# Rust binary
RUST_TARGET := target/release/litmus

# Go binary
GO_COLLECTOR := cmd/collect-benign/collect-benign

all: build

# Build Rust CLI
build:
	cargo build --release

# Build Go collector
collector: $(GO_COLLECTOR)

$(GO_COLLECTOR): cmd/collect-benign/main.go
	cd cmd/collect-benign && go build -o collect-benign .

# Python virtual environment
venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: training/requirements.txt
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r training/requirements.txt
	touch $(VENV_DIR)/bin/activate

# Download benign samples
data: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS)

# Download from specific sources
data-pypi: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources pypi

data-npm: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources npm

data-cargo: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources cargo

data-rubygems: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources rubygems

data-go: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources go

data-wolfi: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources wolfi

data-fedora: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources fedora

data-debian: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources debian

data-freebsd: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources freebsd

data-openbsd: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources openbsd

data-netbsd: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources netbsd

data-bsd: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources freebsd,openbsd,netbsd

data-gnu: collector
	./$(GO_COLLECTOR) --output $(DATA_DIR) --max $(MAX_PACKAGES) --workers $(WORKERS) --sources freebsd-ports

# Build cleave if needed
cleave: $(CLEAVE_BIN)

$(CLEAVE_BIN):
	cd $(CLEAVE_DIR) && cargo build --release

# Symlink cleave traits for capability detection
traits:
	@test -L traits || ln -sf $(CLEAVE_DIR)/traits traits

# Extract features using fast Rust extractor (default)
features: build traits
	@mkdir -p $(FEATURES_DIR)
	@echo "Extracting benign features..."
	find $(DATA_DIR) -type f ! -name "*.json" ! -name "*.txt" ! -name ".*" 2>/dev/null | \
		./target/release/litmus extract - \
			-o $(FEATURES_DIR)/benign.json \
			-l 0 \
			-w $(WORKERS) \
			$(if $(filter-out 0,$(LIMIT)),--limit $(LIMIT),)
	@echo "Extracting malicious features..."
	find $(MALWARE_DIR) -type f ! -name "*.json" ! -name "*.txt" ! -name ".*" 2>/dev/null | \
		./target/release/litmus extract - \
			-o $(FEATURES_DIR)/malware.json \
			-l 1 \
			-w $(WORKERS) \
			$(if $(filter-out 0,$(LIMIT)),--limit $(LIMIT),)

# Batch size for features-batched (default 1024)
BATCH_SIZE ?= 1024

# Extract features in batches (crash-resilient)
# Use EXTENSION=js to filter by file extension
features-batched: build traits
	@mkdir -p $(FEATURES_DIR)/batches
	@echo "Extracting benign features in batches of $(BATCH_SIZE)..."
	@./scripts/batch-extract.sh \
		"$(DATA_DIR)" \
		"$(FEATURES_DIR)/batches/benign" \
		0 \
		$(BATCH_SIZE) \
		$(WORKERS) \
		$(LIMIT) \
		$(if $(EXTENSION),--extension $(EXTENSION),)
	@echo "Merging benign batches..."
	@$(PYTHON) scripts/merge-batches.py $(FEATURES_DIR)/batches/benign $(FEATURES_DIR)/benign.json
	@echo "Extracting malicious features in batches of $(BATCH_SIZE)..."
	@./scripts/batch-extract.sh \
		"$(MALWARE_DIR)" \
		"$(FEATURES_DIR)/batches/malware" \
		1 \
		$(BATCH_SIZE) \
		$(WORKERS) \
		$(LIMIT) \
		--require-suspicious \
		$(if $(EXTENSION),--extension $(EXTENSION),)
	@echo "Merging malware batches..."
	@$(PYTHON) scripts/merge-batches.py $(FEATURES_DIR)/batches/malware $(FEATURES_DIR)/malware.json
	@echo "Batch extraction complete!"

# Extract features using Python (slower, but more flexible)
features-python: venv cleave
	@mkdir -p $(FEATURES_DIR)
	$(PYTHON) training/extract_features.py \
		--benign $(DATA_DIR) \
		--malicious $(MALWARE_DIR) \
		--output $(FEATURES_DIR)/features.npz \
		--cache $(CACHE_DIR) \
		--cleave-dir $(CLEAVE_DIR) \
		$(if $(filter-out 0,$(LIMIT)),--max-samples $(LIMIT),) \
		$(if $(EXTENSIONS),--extensions $(EXTENSIONS),)

# Train XGBoost model from fast-extracted JSON features
train: venv
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) training/train_from_json.py \
		--benign $(FEATURES_DIR)/benign.json \
		--malware $(FEATURES_DIR)/malware.json \
		--output $(MODEL_DIR)/litmus_v1.json \
		--features-npz $(FEATURES_DIR)/features.npz

# Train per-filetype models (recommended for mixed datasets)
train-per-filetype: venv
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) training/train_per_filetype.py \
		--benign $(FEATURES_DIR)/benign.json \
		--malware $(FEATURES_DIR)/malware.json \
		--output-dir $(MODEL_DIR) \
		--min-samples 50

# Train from Python-extracted NPZ features
train-npz: venv
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) training/train_model.py \
		--input $(FEATURES_DIR)/features.npz \
		--output $(MODEL_DIR)/litmus_v1.json

# Evaluate model
evaluate: venv
	$(PYTHON) training/evaluate.py \
		--features $(FEATURES_DIR)/features.npz \
		--model $(MODEL_DIR)/litmus_v1.json

# Evaluate against held-out test set (NOT used for training)
test-eval: build venv
	$(PYTHON) training/evaluate_test.py \
		--test-dir $(TEST_DIR) \
		--litmus ./target/release/litmus \
		--output evaluation/test_report.json

# Predict classification for a file (usage: make predict FILE=/path/to/sample)
predict: venv cleave
	$(PYTHON) training/predict.py --cleave-dir $(CLEAVE_DIR) $(FILE)

# Run SHAP explanation on a sample (usage: make explain FILE=/path/to/sample)
explain: venv cleave
	$(PYTHON) training/predict.py --cleave-dir $(CLEAVE_DIR) --explain $(FILE)

# Full pipeline (assumes data already collected)
pipeline: features train evaluate test-eval

# Full pipeline with batch extraction (crash-resilient, assumes data already collected)
pipeline-batched: features-batched train evaluate test-eval

# Full pipeline including data collection (for initial setup)
pipeline-with-data: data features train evaluate test-eval

# Full pipeline with data collection and batch extraction
pipeline-batched-with-data: data features-batched train evaluate test-eval

# Lint and test
lint:
	cargo clippy -- -D warnings
	cd cmd/collect-benign && go vet ./...

test:
	cargo test
	cd cmd/collect-benign && go test ./...

# Clean build artifacts
clean:
	cargo clean
	rm -f $(GO_COLLECTOR)
	rm -rf $(VENV_DIR)
	rm -rf $(FEATURES_DIR)

# Clean cleave cache (forces re-analysis)
clean-cache:
	rm -rf $(CACHE_DIR)

# Deep clean (including downloaded data and cache)
clean-all: clean clean-cache
	rm -rf $(DATA_DIR)

# Show help
help:
	@echo "litmus - Malware Classification Tool"
	@echo ""
	@echo "Data Collection:"
	@echo "  make collector     Build the Go sample collector"
	@echo "  make data          Download benign samples from all sources"
	@echo "  make data-pypi     Download from PyPI only"
	@echo "  make data-npm      Download from NPM only"
	@echo "  make data-cargo    Download from Cargo only"
	@echo "  make data-rubygems Download from RubyGems only"
	@echo "  make data-go       Download from Go proxy only"
	@echo "  make data-wolfi    Download from Wolfi only"
	@echo "  make data-fedora   Download from Fedora only"
	@echo "  make data-debian   Download from Debian only"
	@echo "  make data-freebsd  Download FreeBSD core sets"
	@echo "  make data-openbsd  Download OpenBSD core sets"
	@echo "  make data-netbsd   Download NetBSD core sets"
	@echo "  make data-bsd      Download all BSD core sets"
	@echo "  make data-gnu      Download GNU source tarballs"
	@echo ""
	@echo "Training:"
	@echo "  make venv          Set up Python virtual environment"
	@echo "  make features      Extract features (fast Rust extractor)"
	@echo "  make features-python  Extract features (slower Python method)"
	@echo "  make train         Train single XGBoost model from JSON features"
	@echo "  make train-per-filetype  Train per-filetype models (recommended)"
	@echo "  make train-npz     Train XGBoost model from NPZ features"
	@echo "  make evaluate      Evaluate model performance"
	@echo "  make test-eval     Evaluate against held-out test set"
	@echo ""
	@echo "Pipelines:"
	@echo "  make pipeline      Run full pipeline (assumes data already collected)"
	@echo "  make pipeline-batched  Full pipeline with batch extraction (crash-resilient)"
	@echo "  make pipeline-with-data  Full pipeline including data collection"
	@echo "  make pipeline-batched-with-data  Full pipeline with data collection and batch extraction"
	@echo ""
	@echo "Build:"
	@echo "  make build         Build Rust CLI"
	@echo "  make lint          Run linters"
	@echo "  make test          Run tests"
	@echo ""
	@echo "Configuration (override with environment variables):"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  MALWARE_DIR=$(MALWARE_DIR)"
	@echo "  TEST_DIR=$(TEST_DIR)"
	@echo "  MAX_PACKAGES=$(MAX_PACKAGES)"
	@echo "  WORKERS=$(WORKERS)"
	@echo "  LIMIT=$(LIMIT) (max samples per class, 0=unlimited)"
	@echo "  EXTENSION=$(EXTENSION) (filter by file extension, e.g., js, py, exe)"
	@echo ""
	@echo "Examples:"
	@echo "  make data MAX_PACKAGES=100       Download 100 packages per source"
	@echo "  make features LIMIT=10000        Extract features (max 10k per class)"
	@echo "  make pipeline LIMIT=5000         Full pipeline with sample limit"
	@echo "  make pipeline EXTENSION=js       Train only on JavaScript files"
	@echo "  make pipeline EXTENSION=py       Train only on Python files"
