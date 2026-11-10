# Datasets

| File | Description |
|------|-------------|
| `student_dropout_uci697.csv` | Original, unmodified file from the UCI Machine Learning Repository (4,424 students, 36 attributes + Target; `;`-separated, UTF-8 with BOM) |
| `data_dictionary.csv` | Every column: original name, short name used in the code, description, group, min and max |
| `sample_new_students.csv` | 25 students in the original layout without the Target column - use it on the dashboard's "Score new students" tab |
| `processed/train.csv`, `processed/test.csv` | Stratified 80/20 split (seed 42) created by `run_pipeline.py` |
| `processed/predictions_<checkpoint>.csv` | Probabilities, risk tier and top reasons for the 885 test students at each checkpoint |

**Source and licence:** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
*Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5MC89 - licensed under CC BY 4.0. Described in Realinho et al.,
*Data* 7(11):146, 2022, https://doi.org/10.3390/data7110146.

Students are identified only by a row number; the file contains no names or contact details.
