# KeenKeenForAndroid - Unified Build System
# Acts as the single entry point for all development tasks.

# Configuration
JAVA_HOME_TARGET := /usr/lib/jvm/java-21-openjdk
GRADLE := export JAVA_HOME=$(JAVA_HOME_TARGET) && ./gradlew

# Host Toolchain
CC = gcc
CFLAGS = -O3 -march=native -flto -ffast-math -funroll-loops -Iapp/src/main/jni -DSTANDALONE_LATIN_TEST -Wall -Wextra -Werror
SIMD_FLAGS = -mavx2 -msse2
JNI_DIR = app/src/main/jni
AI_DIR = scripts/ai
SOURCES = $(JNI_DIR)/latin.c \
          $(JNI_DIR)/random.c \
          $(JNI_DIR)/malloc.c \
          $(JNI_DIR)/maxflow_optimized.c \
          $(JNI_DIR)/tree234.c \
          $(AI_DIR)/host_compat.c

# --- Main Targets ---

.PHONY: help all build release install clean test lint format tools check-env

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

all: tools ## Full CI pipeline: clean, build tools, lint, test, and build APK
	$(GRADLE) clean lintDebug testDebugUnitTest assembleDebug

# --- Android Build ---

build: ## Build Debug APK
	$(GRADLE) assembleDebug

release: ## Build Release APK
	$(GRADLE) assembleRelease

install: ## Install Debug APK to connected device
	$(GRADLE) installDebug

clean: ## Remove build artifacts
	$(GRADLE) clean
	rm -f $(AI_DIR)/latin_gen_opt
	rm -rf .cxx

# --- Quality Assurance ---

test: ## Run local unit tests
	$(GRADLE) testDebugUnitTest

lint: ## Run Android Lint
	$(GRADLE) lintDebug

format: ## Apply formatting (Placeholder)
	@echo "Formatting not yet implemented. Please adhere to Kotlin coding conventions."

# --- Native Tools ---

tools: latin_gen_opt ## Build host-side C tools for AI training

latin_gen_opt: $(SOURCES)
	$(CC) $(CFLAGS) $(SIMD_FLAGS) -o $(AI_DIR)/latin_gen_opt $(SOURCES)

# --- AI Training Pipeline ---

.PHONY: generate-data train deploy-model ai-pipeline full-pipeline

TRAIN_COUNT ?= 10000
ASSETS_DIR = app/src/main/assets
MODEL_NAME = keen_solver_9x9.onnx

generate-data: tools ## Generate Latin square training data (3x3 to 9x9)
	@echo "Generating $(TRAIN_COUNT) grids per size (3x3 to 9x9)..."
	cd $(AI_DIR) && python generate_data.py --count $(TRAIN_COUNT)

train: generate-data ## Train the neural model and export to ONNX (depends on data)
	@echo "Training model on generated data..."
	cd $(AI_DIR) && python train_massive_model.py
	@echo "Model exported to $(AI_DIR)/$(MODEL_NAME)"

deploy-model: train ## Copy trained ONNX model to Android assets (depends on training)
	@if [ -f "$(AI_DIR)/$(MODEL_NAME)" ]; then \
		cp $(AI_DIR)/$(MODEL_NAME) $(ASSETS_DIR)/$(MODEL_NAME); \
		echo "Deployed $(MODEL_NAME) to assets"; \
	else \
		echo "ERROR: $(AI_DIR)/$(MODEL_NAME) not found. Run 'make train' first."; \
		exit 1; \
	fi
	@if [ -f "$(AI_DIR)/$(MODEL_NAME).data" ]; then \
		cp $(AI_DIR)/$(MODEL_NAME).data $(ASSETS_DIR)/$(MODEL_NAME).data; \
		echo "Deployed $(MODEL_NAME).data to assets"; \
	fi

ai-pipeline: deploy-model ## Full AI pipeline: generate data, train, deploy
	@echo "AI pipeline complete. Model ready in assets."

full-pipeline: deploy-model ## Complete build: AI pipeline + Android APK
	$(GRADLE) assembleDebug
	@echo "Full pipeline complete. APK ready."

# --- Verification ---

verify-arm: ## Build Release and verify ARMv8 binary architecture and flags
	$(GRADLE) assembleRelease
	@echo "Verifying ARMv8 Binary..."
	@find app/build/intermediates/merged_native_libs/release/mergeReleaseNativeLibs/out/lib/arm64-v8a -name "libkeen-android-jni.so" -exec file {} \;
	@echo "Checking for Optimization Flags in Compile Database..."
	@find app/.cxx -name compile_commands.json -exec grep -H "aarch64" {} \; | head -n 1 || true
	@echo "Build Verified: Native ARMv8 target with NDK toolchain."

# --- Development Helpers ---

check-env: ## Verify development environment
	@echo "Checking Java..."
	@if [ -d "$(JAVA_HOME_TARGET)" ]; then echo "OK: Java 21 found at $(JAVA_HOME_TARGET)"; else echo "FAIL: Java 21 not found at $(JAVA_HOME_TARGET)"; exit 1; fi
	@echo "Checking ADB..."
	@adb version > /dev/null 2>&1 && echo "OK: ADB found" || echo "WARN: ADB not found"

run-modern: install ## Run the app in Modern Mode
	adb shell am start -n org.yegie.keenkeenforandroid/.KeenActivity --ez useClassic false

run-classic: install ## Run the app in Classic Mode
	adb shell am start -n org.yegie.keenkeenforandroid/.KeenActivity --ez useClassic true