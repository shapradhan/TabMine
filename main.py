from dotenv import load_dotenv
from os import getenv
if __name__ == '__main__':
    load_dotenv()
    
    DB_CONFIG_FILE = getenv('DB_CONFIG_FILE')
    SECTION = getenv('SECTION') 
