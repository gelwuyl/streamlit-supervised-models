# su-ntu-ctp/5m-data-3.4-supervised-learning-advanced - assignment

Notebook (graded deliverable) + interactive Streamlit playground for the regularization & ensemble assignment. Both share the same `src/` engine, so the numbers always match.

## Run the notebook (validate the assignment first)
1. `pip install -r requirements.txt`
2. Open `notebooks/assignment.ipynb` and **Run All**.
3. Confirm: 3 regression MSEs print, 3 classifiers show F1 + AUC, and the
   coefficient-path + ROC charts render.

## Run the app locally
```
streamlit run streamlit_app.py
```

## Deploy (later — not required to validate)
Push to a new GitHub repo, then create a New app on Streamlit Community Cloud pointing at `streamlit_app.py`.

## Credits
- scikit-learn (`diabetes`, `breast_cancer` datasets)