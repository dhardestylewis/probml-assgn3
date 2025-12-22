
import urllib.request
import os

def download_file(url, filename):
    print(f"Attempting to download {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = response.read()
            if len(data) > 1000: # Ensure it's not a tiny error page
                with open(filename, 'wb') as out_file:
                    out_file.write(data)
                print(f"Successfully downloaded {filename} ({len(data)} bytes)")
            else:
                print(f"Warning: Downloaded file too small ({len(data)} bytes), likely an error page.")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

os.makedirs('hws/hw3.d/images', exist_ok=True)

# Columbia University Shield (New URL)
# Using a simpler png source often indexed
url_cu = "https://logos-world.net/wp-content/uploads/2020/12/Columbia-University-Logo.png"
download_file(url_cu, 'hws/hw3.d/images/columbia_logo.png')

# Columbia Engineering (Using fallback if not found, let's try a university page or wikimedia archive)
# This link is often stable for SEAS
url_seas = "https://upload.wikimedia.org/wikipedia/en/2/27/Columbia_Engineering_Logo.png"
download_file(url_seas, 'hws/hw3.d/images/columbia_engineering_logo.png')
