.PHONY: run_und generate_und_violin_plots

all: run_models generate_violin_plots analyze_breadth analyze_depth

und: run_und generate_und_violin_plots analyze_und_breadth analyze_und_depth

inc: run_inc generate_inc_violin_plots analyze_inc_breadth analyze_inc_depth

#--------------------------------------------------------------------------------------------------

# Commands for running both proofs of concept.
run_models: run_und run_inc

generate_violin_plots: generate_und_violin_plots generate_inc_violin_plots

analyze_breadth: analyze_und_breadth analyze_inc_breadth
	python analyze_breadth.py both map

analyze_depth: analyze_und_depth analyze_inc_depth

#--------------------------------------------------------------------------------------------------

# Commands for running the first proof of concept.
run_und: und_results
	python run_models.py und clip
	python run_models.py und radio
	python run_models.py und siglip
	python run_models.py und siglip2

generate_und_violin_plots: und_analysis
	python generate_violin_plots.py und clip
	python generate_violin_plots.py und radio
	python generate_violin_plots.py und siglip
	python generate_violin_plots.py und siglip2

analyze_und_breadth: und_analysis
	python analyze_breadth.py und map
	python analyze_breadth.py und plot

analyze_und_depth: und_analysis
	python analyze_depth.py und clip
	python analyze_depth.py und clip-quickgelu
	python analyze_depth.py und clip_and_clip-quickgelu
	python analyze_depth.py und siglip
	python analyze_depth.py und siglip2
	python analyze_depth.py und radio

#--------------------------------------------------------------------------------------------------

# Commands for running the second proof of concept.
run_inc: inc_results
	python run_models.py inc clip
	python run_models.py inc radio
	python run_models.py inc siglip
	python run_models.py inc siglip2

generate_inc_violin_plots: inc_analysis
	python generate_violin_plots.py inc clip
	python generate_violin_plots.py inc radio
	python generate_violin_plots.py inc siglip
	python generate_violin_plots.py inc siglip2

analyze_inc_breadth: inc_analysis
	python analyze_breadth.py inc map
	python analyze_breadth.py inc plot

analyze_inc_depth: inc_analysis
	python analyze_depth.py inc clip
	python analyze_depth.py inc clip-quickgelu
	python analyze_depth.py inc clip_and_clip-quickgelu
	python analyze_depth.py inc siglip
	python analyze_depth.py inc siglip2
	python analyze_depth.py inc radio

#--------------------------------------------------------------------------------------------------

# check and create und_results directories if nonexistent
und_results:
	mkdir -p und_results/CLIP_results
	mkdir -p und_results/SigLIP_results
	mkdir -p und_results/SigLIP2_results
	mkdir -p und_results/RADIO_results

# check and create und_analysis directories if nonexistent
und_analysis:
	mkdir -p und_analysis/clip
	mkdir -p und_analysis/siglip
	mkdir -p und_analysis/siglip2
	mkdir -p und_analysis/radio

# check and create inc_results directories if nonexistent
inc_results:
	mkdir -p inc_results/CLIP_results
	mkdir -p inc_results/SigLIP_results
	mkdir -p inc_results/SigLIP2_results
	mkdir -p inc_results/RADIO_results

# check and create inc_analysis directories if nonexistent
inc_analysis:
	mkdir -p inc_analysis/clip
	mkdir -p inc_analysis/siglip
	mkdir -p inc_analysis/siglip2
	mkdir -p inc_analysis/radio

#--------------------------------------------------------------------------------------------------

# Removes all results and analysis files.
clean:
	rm -f und_results/CLIP_results/*.csv
	rm -f und_results/SigLIP_results/*.csv
	rm -f und_results/SigLIP2_results/*.csv
	# rm -f und_results/RADIO_results/*.csv # [COMMENTED OUT], confirm and then uncomment to delete
	rm -f und_analysis/clip/*.png
	rm -f und_analysis/siglip/*.png
	rm -f und_analysis/siglip2/*.png
	# rm -f und_analysis/radio/*.png # [COMMENTED OUT], confirm and then uncomment to delete
	rm -f und_analysis/*.csv
	rm -f und_analysis/*.png
	rm -f inc_results/CLIP_results/*.csv
	rm -f inc_results/SigLIP_results/*.csv
	rm -f inc_results/SigLIP2_results/*.csv
	# rm -f inc_results/RADIO_results/*.csv # [COMMENTED OUT], confirm and then uncomment to delete
	rm -f inc_analysis/clip/*.png
	rm -f inc_analysis/siglip/*.png
	rm -f inc_analysis/siglip2/*.png
	# rm -f inc_analysis/radio/*.png # [COMMENTED OUT], confirm and then uncomment to delete
	rm -f inc_analysis/*.csv
	rm -f inc_analysis/*.png
	rm -f both_analysis/*.csv
	rm -f both_analysis/*.png