.PHONY: run_models analyze_models

all: run_models analyze_models

run_models: SigLIP_results SigLIP2_results RADIO_results
	python run_models.py radio
	python run_models.py siglip
	python run_models.py siglip2

analyze_models: analysis/siglip analysis/siglip2 analysis/radio
	python analyze_models.py radio
	python analyze_models.py siglip
	python analyze_models.py siglip2

# check and create results directories if nonexistent
SigLIP_results:
	mkdir -p SigLIP_results

SigLIP2_results:
	mkdir -p SigLIP2_results

RADIO_results:
	mkdir -p RADIO_results

# check and create analysis directories if nonexistent
analysis/siglip:
	mkdir -p analysis/siglip

analysis/siglip2:
	mkdir -p analysis/siglip2

analysis/radio:
	mkdir -p analysis/radio

clean:
	rm -f SigLIP_results/*.csv
	rm -f SigLIP2_results/*.csv
	rm -f RADIO_results/*.csv
	rm -f analysis/siglip/*.png
	rm -f analysis/siglip2/*.png
	rm -f analysis/radio/*.png