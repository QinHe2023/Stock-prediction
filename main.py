import streamlit as st
import pandas as pd
import yfinance as yf
from authenticate import Authenticate
from model_google import predict_google
from model_apple import predict_apple
from model_google_predict_one_day import predict_google_one_day
from model_apple_predict_one_day import predict_apple_one_day
import matplotlib.pyplot as plt


#st.title(" Welcome to ","<span style='color: red;'> IntelliStock </span> ",unsafe_allow_html=True) # the webpage title
st.markdown("<h1 > Welcome to  <span style='color: red;'> IntelliStock</span></h1>",unsafe_allow_html=True)

st.write("""
        
            **Empowering Investments, Illuminating Futures.**
            """) # the slogan of the webpage
local_image_path = "image.jpg"
st.image(local_image_path, use_column_width=True)
def creat_initial_users():  # define a function to create all the initial users' name and password for user login 
    credential_dict = {} # create an empty dictionary to save user name and password
    try:
        file_path = 'credentials.txt'
        
        with open(file_path, 'r',encoding='utf-8') as file:
                user_info = file.read() # read the credentials file
            
        for user in user_info.split('\n'):
            credential_dict[user.split(',')[0]] = user.split(',')[1]   # # save user name and password in a dicitonary, in the form of {'username':'password','username':'password',..}
            
    except FileNotFoundError:
        return "File not found."
    
    return credential_dict # return the dictionary which saved the usernames and passwords
    
credentials = creat_initial_users() # call the function to create all the initial usernams and passwords 
#print(credentials.keys())

authenticator = Authenticate(credentials) # create an Object to manage the user login, logout, and register 

if 'authentication_status' not in st.session_state:  # user st.session_state to save the username and authentication status
    st.session_state['authentication_status'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
#if 'login' not in st.session_state:
#    st.session_state['login'] = False

    
def main():

   
    st.markdown(" <h3 style='text-align: center;'>User Login/Register </h3> ",unsafe_allow_html=True)
    choice = st.selectbox('Login/Register',['Login','Register'])
    if choice =='Login':
        print(authenticator.credentials)

        with st.form("login"):
            username = st.text_input("Username:")
            paswd = st.text_input("Password:",type="password")
            submit_state = st.form_submit_button("Login")
            if submit_state:
                if username == '' or paswd == '': # identify if user input all fields
                    st.warning("Please input all the fields")
                else:
                    authenticator.login(username,paswd) # the object authenticator will implement the user authentication 
                        

        if st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')

        elif st.session_state["authentication_status"] is True:
            st.success("Logged in successfully!")
             # show user's name if login sucessful
            st.markdown(f'<h3>Hello,{st.session_state["username"]} </h3> ',unsafe_allow_html=True)
            if st.button("Logout"):
                authenticator.logout()
                st.success("Logged out successfully!")
                
            
            

            #st.set_page_config()

            search_query = st.sidebar.text_input(f"Search for a stock: ","AAPL") # user input the symbol of the stock to search 
        
            #matching_symbols = yf.download(search_query)
            #print(matching_symbols)
            tickerSymbol = search_query
            tickerData = yf.Ticker(tickerSymbol) # get the stock's data
            if tickerData is not None:

                # Get information about the stock
                info = tickerData.info
                #print(info)

                # Display stock information
                if info:
                    st.write(f"**Stock Information for {tickerSymbol}**")
                    info_df = pd.DataFrame([info])
                    st.dataframe(info_df) 
                else:
                    st.write(f"No information found for {tickerSymbol}") # give feedback if no stock found
            else:
                st.write(f"No information found for {tickerSymbol}")


            start_date = st.sidebar.date_input("Start date: ") 
            end_date = st.sidebar.date_input("End date: ")# allow user to select the start date and end date to see the price of the stock
            period_options = ["One Day", "One Week","One month","One Year"] 
            interval = st.sidebar.radio("Interval:", period_options)
            #interval = st.radio("Interval:", period_options, format_func=lambda x: f'<div style="display:inline-block; margin-right:20px;">{x}</div>', key="period_options")
            tickerDf = tickerData.history(period="1d",start=start_date,end=end_date)
            #print(tickerDf)
            #print(tickerDf.Close)
            st.write("You selected:", start_date,'-',end_date)

            st.write("Closing price")
            st.line_chart(tickerDf["Close"]) # visualize the historic data
            st.write("Volume")
            st.bar_chart(tickerDf.Volume) # visualize the historic data
        
        

            st.sidebar.write(f"**Stock Prediction**")
            stock_to_predict = ['AAPL','GOOG','NFLX']
            selected_option = st.sidebar.selectbox("Select an option:", stock_to_predict)
            st.write(f"**Stock Prediction**")
            st.write("You selected:", selected_option)
            #start_date_predict = st.date_input("Start date: ",key="predict_start") 
            #end_date_predict = st.date_input("End date: ",key="predict-end")# allow user to select the start date and end date to predict the price based on the choosen period
            #predict_data = yf.Ticker(selected_option) #get the data of the selected stock
            #predict_data_period = predict_data.history(period="1d",start=start_date_predict,end=end_date_predict) # get the time period data
            #st.line_chart(predict_data_period["Close"]) # show the close price in the time period
            #print(predict_data_period["Close"])
            #print(type(predict_data_period["Close"]))
            
            #predict ten days
            real_price,predict_price,stock_time,model,last_ten_days = predict_apple()
            
            predict_price_reshape = predict_price.reshape(-1) # transform the 2-D narry to 1-D narry, for later chaning to dataframe format
            last_ten_days_predict_price = model.predict (last_ten_days) # the predict prices of the last ten days
            last_ten_days_predict_price_reshape = last_ten_days_predict_price.reshape(-1)
            st.write(f"*Ten days prediction*")
                        
            df = pd.DataFrame({
                'stock_time': stock_time,
                'predict_price':predict_price_reshape,
                'real_price': real_price
            
            })
            df.set_index('stock_time', inplace=True)
            
            st.write("The hisotric comparison of real price and predicted price:")
           
            st.line_chart(df)
            
            st.write("The future 10 days' predicting prices:")
            df2 = pd.DataFrame({
                
                'Predict_price':last_ten_days_predict_price_reshape
            })
            
            st.line_chart(df2)
            st.write("The following day predited price:", last_ten_days_predict_price_reshape)
            
            st.write("One day prediction")
            
            # predict oneday 
            real_price_one_day,predict_price_one_day,stock_time_one_day,model_one_day,last_one_day = predict_apple_one_day()
                        
            predict_price_one_day_reshape = predict_price_one_day.reshape(-1) # transform the 2-D narry to 1-D narry, for later chaning to dataframe format
            last_one_day_predict_price = model_one_day.predict (last_one_day) # the predict prices of the last ten days
            last_one_day_predict_price_reshape = last_one_day_predict_price.reshape(-1)
                         
            df_one_day = pd.DataFrame({
                'stock_time': stock_time_one_day,
                'predict_price':predict_price_one_day_reshape,
                'real_price': real_price_one_day
            
            })
            df_one_day.set_index('stock_time', inplace=True)
            st.write("The hisotric comparison of real price and predicted price:")
        
            st.line_chart(df_one_day)
            # df_one_day_2 = pd.DataFrame({
                
            #     'Predict_price':last_one_day_predict_price_reshape
            # })
            # st.line_chart(df_one_day_2)
            
            st.write("The following day predited price:", last_one_day_predict_price_reshape)
            print(last_one_day_predict_price_reshape)
            
            
    else:
        
        with st.form("register"):
            username = st.text_input("Username:")
            paswd = st.text_input("Password:",type="password")
            submit_state = st.form_submit_button("Register")
            if submit_state:
                if username == '' or paswd == '': # identify if user input all fields
                    st.warning("Please input all the fields")
                else:
                    result = authenticator.register(username,paswd)
                    if result:
                        st.success("Register successfully!")
                        print(authenticator.credentials)
                        st.balloons()
                    else:
                        st.error("User already exist!")
        
        
            

if __name__ == "__main__":
    main()