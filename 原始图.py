#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      14014
#
# Created:     06.04.2019
# Copyright:   (c) 14014 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    pass

if __name__ == '__main__':
    main()



# Import data
data = pd.read_csv('data_stocks.csv')
# Drop date variable
x=data['DATE']
y=data['SP500']
# Make data a numpy array
x=x.values
y=y.values

plt.plot(x,y)
plt.show()
