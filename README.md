**GOAL**
Machine learning on crypto currency markets. I'd like to develop a model to accurately predict when a crypto market will increase in value.

 **PLAN**
- retrieve crypto market data from kraken api.
- develop/train *logistic regression* model using training data.
- develop/train *neural network* model using training data.
- develop/train *ad hoc* model using training data.
- develop function that uses model to determine if the crypto is likely increase in value
- back test function on test data
- develop trade management program to automate buying /selling

 **STARTING THE PROGRAM**
 1.  Create a python virtual environment in the "kraken" folder using
 `py -m venv env`

 2.  Run virtual environment activation script
 `env/Scripts/activate`

3. Install depedencies
`pip install -r requirements.txt`

4. Run program
`py gather_data.py`


**REFERENCES**
https://algotrading101.com/learn/kraken-api-guide/
