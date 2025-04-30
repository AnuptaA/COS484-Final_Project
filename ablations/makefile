.PHONY: run_models analyze_models

all: run_models analyze_models

run_models: CLIP_results SigLIP_results SigLIP2_results RADIO_results
	python run_models.py clip
	python run_models.py radio
	python run_models.py siglip
	python run_models.py siglip2

analyze_models: analysis/clip analysis/siglip analysis/siglip2 analysis/radio
	python analyze_models.py clip
	python analyze_models.py radio
	python analyze_models.py siglip
	python analyze_models.py siglip2

# check and create results directories if nonexistent
CLIP_results:
	mkdir -p CLIP_results

SigLIP_results:
	mkdir -p SigLIP_results

SigLIP2_results:
	mkdir -p SigLIP2_results

RADIO_results:
	mkdir -p RADIO_results

# check and create analysis directories if nonexistent
analysis/clip:
	mkdir -p analysis/clip

analysis/siglip:
	mkdir -p analysis/siglip

analysis/siglip2:
	mkdir -p analysis/siglip2

analysis/radio:
	mkdir -p analysis/radio

clean:
	rm -f CLIP_results/*.csv
	rm -f SigLIP_results/*.csv
	rm -f SigLIP2_results/*.csv
	rm -f RADIO_results/*.csv
	rm -f analysis/clip/*.png
	rm -f analysis/siglip/*.png
	rm -f analysis/siglip2/*.png
	rm -f analysis/radio/*.png