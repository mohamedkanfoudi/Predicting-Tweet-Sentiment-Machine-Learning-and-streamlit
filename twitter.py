
import tweepy
import csv #Import csv
auth = tweepy.auth.OAuthHandler('NzvB2AJxTZ33ljb6rxXlcrRFm', 'DUSJgUIYciLojp1S0vs9QTi2uAaVjkni4HOCCa5xEW3hz8OI5e')
auth.set_access_token('320858460-oU8u0g7O2Vk4qF6Rk6aMiPYoOmjyIVvHHCXuBvIM', '2sfZfyaSi1eHqZOE4cQ2eJNso4EqeDMorZwJWr3aDH0me')

api = tweepy.API(auth)

# Open/create a file to append data to
csvFile = open('result.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

requete = "Covid-19 OR Covid OR Corona OR Pandémie OR épidémie OR Coronavirus OR virus"

for tweet in tweepy.Cursor(api.search,
                           q = requete,
                           since = "2021-11-12",
                           until = "2021-11-13",
                           lang = "en").items():

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.id,tweet.created_at, tweet.text.encode('utf-8')])
    print (tweet.id,tweet.created_at, tweet.text.encode('utf-8') )
csvFile.close()
