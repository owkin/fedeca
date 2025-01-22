python plot_kms.py -R fedeca_dataset/fedeca_dataset/fl/\(\'ffcd\',\ \'pancan\'\)_results_2025-01-17_15-10-22_is_folfirinoxTrue_treatmentffcd_fed_kaplan_120b1807-5f28-4f26-a4aa-eddd68847dd1.pkl  2>&1 | tee -a log2
python plot_kms.py -R fedeca_dataset/fedeca_dataset/fl/\(\'ffcd\',\ \'pancan\'\)_results_2025-01-17_21-53-51_is_folfirinoxFalse_treatmentffcd_fed_kaplan_d110ed36-7a95-434f-9a60-e4f6b1484e65.pkl  2>&1 | tee -a log3
python plot_kms.py -R fedeca_dataset/fedeca_dataset/fl/\(\'ffcd\',\ \'idibigi\'\)_results_2025-01-19_18-40-01_is_folfirinoxFalse_treatmentidibigi_fed_kaplan_d21d92ba-1dc1-4940-8dee-8133da8c9008.pkl  2>&1 | tee -a log1

python plot_kms.py -R fedeca_dataset/fedeca_dataset/fl/\(\'idibigi\',\ \'ffcd\'\)_results_2025-01-17_11-58-55_is_folfirinoxTrue_treatmentidibigi_fed_kaplan_d6e19450-49f7-4d11-a759-7d281048c637.pkl  2>&1 | tee -a log4
python plot_kms.py -R fedeca_dataset/fedeca_dataset/fl/\(\'idibigi\',\ \'pancan\'\)_results_2025-01-17_10-20-54_is_folfirinoxTrue_treatmentidibigi_fed_kaplan_c605b601-1697-46cd-86be-0a5f1e574c04.pkl  2>&1 | tee -a log5
python plot_kms.py -R fedeca_dataset/fedeca_dataset/fl/\(\'pancan\',\ \'idibigi\'\)_results_2025-01-19_20-21-26_is_folfirinoxFalse_treatmentidibigi_fed_kaplan_a7e82932-ee64-469d-8b5c-34a856eebe75.pkl  2>&1 | tee -a log6
