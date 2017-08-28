# higgs

Application of tensorflow on higgs data. Based on:

https://indico.cern.ch/event/433556/contributions/1930588/attachments/1231746/1806025/tensorflow_introduction.pdf

Run step by step:
1. Clone repository
2. You need to download higgs data (HIGGS.csv.gz) and put it to directory ./data/ Download is possible using downloadHiggs.sh or you can do it manually.
3. Run main.py using: 'python main.py low' or 'python main.py high'. Low stands for the low level (21 features) features from dataset, high for high leve features (7 features).
4. Optionally run 'python plot_all.py' to plot everything in pdf file.

In Configuration.py there's 'HIGGS_FRACS' array. It says what fractions of higgs data are going to be used for learning and evaluating. Each run of main.py will loop over HIGGS_FRACS array. For the first run subsets of HIGGS dataset will be created and saved into data as .npy files. As a consequence, first run for given fraction requires more RAM memory.

Results are saved in results_low and results_high. Script saves dictionaries with results but also plots all resutls seperately. In order to plot everything to one file so as to compare side by side you can run 'python plot_all.py'. It uses Configuration.py file with higgs_frac array.
