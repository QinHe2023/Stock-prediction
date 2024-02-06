import streamlit as st

class Authenticate: # create a class to manage the user login, logout and register
    def __init__(self, credentials: dict):
        self.credentials = credentials
        
   
    def login(self,username,pw):
        if username in self.credentials.keys():
            if self.credentials[username] == pw:
                st.session_state['authentication_status'] = True
                #st.session_state['login'] = True
                st.session_state['username'] = username
            else: 
                st.session_state['authentication_status'] = False
        else:
            st.session_state['authentication_status'] = False
                

    def logout(self):
        #st.session_state['login'] = False
        st.session_state['username'] = None
        st.session_state['authentication_status'] = None
    
 
    def register(self,username,pw):
        if username in self.credentials.keys():
            return False
        else:
            with open('credentials.txt', 'a') as file:
                # Write words to the file
                file.write('\n'+username +','+ pw)
                        
            #self.credentials[username] = pw
            return True
        
    
            
        