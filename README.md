# CrossSectionPY
Re-writing the CrossSection repo into python. 
https://github.com/OpenSourceAP/CrossSection

Make sure correct .csv files are in signals/data. For reference, look at the quickrunlist var in globals.py and download them from this bottom link.
https://drive.google.com/drive/folders/1rIJjwi4327ELA6KRuBmP4jvgUf_m9rWr
Make sure you know which are Predictor Portfolios and which ones are Placebo Portfolios

Make sure Comparison_to_Meta_Replications.csv is in root
Make sure Comparison_to_HLZ is in root

If you want all the placebos/predictors the zip files are here:
https://drive.google.com/drive/folders/1rCXYO4R9KIPytX7cFwCL8UTx2dunm1RC

download the placebos/predicots zip files and unzip them into corresponding Signal/Data/placebos or Signal/Data/predictors folders

NOTE: line 53 in settingAndTools, replace that string literal your path for SignalDoc.csv (should be in root)

To run simply run the main.py file in Portfolios/Code from the project root.
