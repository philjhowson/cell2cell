import pandas as pd
import seaborn as sns

def exploration():

    data = pd.read_csv('data/raw/cell2celltrain.csv')

    pd.set_option('future.no_silent_downcasting', True)

    data.replace({'Yes': 1, 'No' : 0}, inplace = True)
    data['CreditRating'].replace({'1-Highest' : 1,
                                  '2-High' : 2,
                                  '3-Good' : 3,
                                  '4-Medium' : 4,
                                  '5-Low' : 5,
                                  '6-VeryLow' : 6,
                                  '7-Lowest': 7}, inplace = True)

    churn = data['Churn'].sum()

    print(f"{churn} customers churned, or {round(churn/len(data) * 100, 2)}% of customers.")

    print(data.info())

    print(set(data['ChildrenInHH']))
    print(set(data['HandsetRefurbished']))
    print(set(data['HandsetWebCapable']))
    print(set(data['TruckOwner']))
    print(set(data['RVOwner']))
    print('Homeownership', set(data['Homeownership']))
    print('BuysViaMailOrder', set(data['BuysViaMailOrder']))
    print('RespondsToMailOffers', set(data['RespondsToMailOffers']))
    print('OptOutMailings', set(data['OptOutMailings']))
    print('NonUSTravel', set(data['NonUSTravel']))
    print('OwnsComputer', set(data['OwnsComputer']))
    print('HasCreditCard', set(data['HasCreditCard']))
    print('NewCellphoneUser', set(data['NewCellphoneUser']))
    
    
    print(set(data['CreditRating']))
    
if __name__ == '__main__':
    exploration()
