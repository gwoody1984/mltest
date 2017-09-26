from pprint import pprint as pp

from datetime import datetime, timedelta, time
import psycopg2
import psycopg2.extras

import pandas as pd

# TODO: extract to config
connection_string = "dbname='glycemiq' user='postgres' password='postgres' host='localhost'"

# TODO: these should come in as a params
user_id = "5WG9G5"
start_date = datetime(2017, 8, 7)
date_range = 4

# TODO: extract to separate consts class
bg_query = """
SELECT calculated_value, 
       to_timestamp(timestamp/1000) as timestamp 
FROM public.bgreadings 
""" #WHERE to_timestamp(timestamp/1000)::date = %s
activity_query = """
SELECT  receive_date,
        date,
        coalesce(steps - lag(steps) over (order by date, receive_date), steps) as steps_increase,
        coalesce(sedentary_minutes - lag(sedentary_minutes) over (order by date, receive_date), sedentary_minutes) as sedentary_minutes_increase,
        coalesce(lightly_active_minutes - lag(lightly_active_minutes) over (order by date, receive_date), lightly_active_minutes) as lightly_active_minutes_increase,
        coalesce(fairly_active_minutes - lag(fairly_active_minutes) over (order by date, receive_date), fairly_active_minutes) as fairly_active_minutes_increase,
        coalesce(very_active_minutes - lag(very_active_minutes) over (order by date, receive_date), very_active_minutes) as very_active_minutes_increase,
        coalesce(calories_out - lag(calories_out) over (order by date, receive_date), calories_out) as calories_out_increase
FROM activity 
WHERE user_id = %s
""" #AND date = %s
food_query = """
SELECT to_timestamp(created/1000) as timestamp,
       glycemicindex,
       calories,
       calcium,
       calciumunit,
       carbs,
       carbsunit,
       fiber,
       fiberunit,
       folate,
       folateunit,
       iron,
       ironunit,
       magnesium,
       magnesiumunit,
       monounsaturatedfat,
       monounsaturatedfatunit,
       niacin,
       niacinunit,
       phosphorus,
       phosphorusunit,
       polyunsaturatedfat,
       polyunsaturatedfatunit,
       potassium,
       potassiumunit,
       protein,
       proteinunit,
       riboflavin,
       riboflavinunit,
       saturatedfat,
       saturatedfatunit,
       sodium,
       sodiumunit,
       sugar,
       sugarunit,
       thiamin,
       thiaminunit,
       totalfat,
       totalfatunit,
       vitamina,
       vitaminaunit,
       vitaminb6,
       vitaminb6unit,
       vitaminc,
       vitamincunit,
       vitamine,
       vitamineunit,
       vitamink,
       vitaminkunit,
       zinc,
       zincunit
FROM public.food
where created is not null
""" #WHERE to_timestamp(created/1000)::date = %s
insulin_query = """
SELECT to_timestamp(created/1000) as timestamp, 
      insulintype,
        unittype,
        units
FROM public.insulindose
""" #WHERE to_timestamp(created/1000)::date = %s
sleep_query = """
SELECT sd.data_point_time,
        sd.level,
        sd.seconds/60 as minutes
FROM public.sleep_detail sd
INNER JOIN public.sleep_summary ss
ON sd.sleep_summary_id = ss.id
WHERE ss.user_id = %s
and sd.level in ('asleep', 'restless')
""" #and ss.date = %s


def get_data() -> dict:
    """
    Gets data from the database

    Note - removes all timezone info from the records -- may need to keep these in future
    """
    data_dict = {}  # container to hold our return data

    # connect to the database
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # bg data
    cur.execute(bg_query)#, (input_date,))
    rows = cur.fetchall()
    bg_data = [(row['timestamp'].replace(tzinfo=None), row['calculated_value']) for row in rows]
    data_dict['bg'] = bg_data

    # activity data
    cur.execute(activity_query, (user_id,))# input_date))
    rows = cur.fetchall()
    data_dict['activity'] = rows

    # food data
    cur.execute(food_query)#, (input_date,))
    rows = cur.fetchall()
    rows = [[row[0].replace(tzinfo=None), *row[1:]] for row in rows]
    data_dict['food'] = rows

    # insulin data
    cur.execute(insulin_query)#, (input_date,))
    rows = cur.fetchall()
    rows = [[row[0].replace(tzinfo=None), *row[1:]] for row in rows]
    data_dict['insulin'] = rows

    # sleep data
    # Note -- not even sure if this is usable. Seems quite fragmented
    cur.execute(sleep_query, (user_id,))#, input_date))
    rows = cur.fetchall()
    data_dict['sleep'] = rows

    return data_dict


def correlate_data(data_dict: dict) -> list:
    """
    Correlates the data in the dictionary to line up with the blood glucose tick intervals

    :rtype: list
    :return: list of tuple containing:

        - bg level timestamp
        - bg level
        - list of changes in activity
            - steps_increase
            - sedentary_minutes_increase
            - lightly_active_minutes_increase
            - fairly_active_minutes_increase
            - very_active_minutes_increase
            - calories_out_increase
        - list of food intake nutrients
            - glycemicindex, calories, calcium, carbs, fiber, folate, iron, magnesium, monounsaturatedfat,
              niacin, phosphorus, polyunsaturatedfat, potassium, protein, riboflavin, saturatedfat, sodium,
              sugar, thiamin, totalfat, vitamina, vitaminb6, vitaminc, vitamine, vitamink, zinc
        - total minutes of sleep
        - current basal insulin over number of hours that it is active
        - bolus insulin at the given time period
    """
    # sort blood glucose data by date
    data_dict['bg'].sort(key=lambda r: r[0])
    bg_data = data_dict['bg']

    correlated_data = []
    for i in range(0, len(bg_data)):
        timestamp, value = bg_data[i]
        five_mins = [item for item in bg_data if item[0] <= timestamp - timedelta(minutes=5)]
        ten_mins = [item for item in bg_data if item[0] <= timestamp - timedelta(minutes=10)]
        fifteen_mins = [item for item in bg_data if item[0] <= timestamp - timedelta(minutes=15)]
        twenty_mins = [item for item in bg_data if item[0] <= timestamp - timedelta(minutes=20)]
        twentyfive_mins = [item for item in bg_data if item[0] <= timestamp - timedelta(minutes=25)]
        thirty_mins = [item for item in bg_data if item[0] <= timestamp - timedelta(minutes=30)]
        label = [item for item in bg_data if
                 timestamp + timedelta(minutes=28) <= item[0] <= timestamp + timedelta(minutes=40)]

        value_list = [label[0][1]] if label else [None]
        value_list.append(value)
        value_list.append(five_mins[-1][1]) if five_mins else value_list.append(None)
        value_list.append(ten_mins[-1][1]) if ten_mins else value_list.append(None)
        value_list.append(fifteen_mins[-1][1]) if fifteen_mins else value_list.append(None)
        value_list.append(twenty_mins[-1][1]) if twenty_mins else value_list.append(None)
        value_list.append(twentyfive_mins[-1][1]) if twentyfive_mins else value_list.append(None)
        value_list.append(thirty_mins[-1][1]) if thirty_mins else value_list.append(None)

        # get change in each respective data set for each new blood glucose value
        activity_change = get_activity_change(bg_data, data_dict['activity'], i, timestamp)
        food_intake = get_food_intake(bg_data, data_dict['food'], i, timestamp)

        # aggregate total sleep for the day at each bg level
        total_sleep = get_sleep(data_dict['sleep'], timestamp)

        # get the current basal and bolus insulin
        basal, bolus = get_insulin(data_dict['insulin'], timestamp)

        correlated_data.append((timestamp, *value_list, *activity_change, *food_intake, total_sleep, basal, bolus))

    return correlated_data


def get_insulin(insulin_data: list, timestamp: datetime) -> tuple:
    """
    Gets the basal rate over the number of hours it is active and any bolus taken in the last 30 minutes
    """
    max_basal_ts = max([insulin[0] for insulin in insulin_data if insulin[0] <= timestamp and insulin[2] == "Basal"])
    tomorrow = datetime.combine(max_basal_ts.date() + timedelta(days=1), time())
    next_basal_ts = min(
        [insulin[0] if insulin[0] > max_basal_ts and insulin[2] == "Basal" else tomorrow for insulin in insulin_data])

    basal = [insulin[3] for insulin in insulin_data if insulin[0] == max_basal_ts and insulin[2] == "Basal"][0]
    basal_rate = basal / ((next_basal_ts - max_basal_ts).seconds / (60 * 60))

    # get bolus
    end_time_window = timestamp + timedelta(minutes=-30)
    bolus = sum([insulin[3] for insulin in insulin_data
                 if timestamp >= insulin[0] > end_time_window and insulin[2] == "Bolus"])

    return basal_rate, bolus


def get_sleep(sleep_data: list, timestamp: datetime) -> int:
    """
    Get the total sleep up to the given timestamp

    Note - may want to look back 24hrs since some sleep would have occurred the night before
    """
    sleep_records = []
    for sleep in sleep_data:
        if sleep[0] <= timestamp:
            sleep_records.append(sleep[2])
    total_sleep = sum(sleep_records)
    return total_sleep


def get_food_intake(bg_data: tuple, food_data: list, i: int, timestamp: datetime) -> list:
    """
    Gets the food intake between ticks of blood glucose.

    Note - this may need to be altered to get food consumed within a given window
     for live predictions
    """
    if i == 0:
        # look for any food consumed on or before the first bg level
        # remove the receive date food[1:]
        foods = [food[1:] for food in food_data if food[0] <= timestamp]
    else:
        # get all food intake between this bg level and the last
        # remove the receive date food[1:]
        last_timestamp, _ = bg_data[i - 1]
        two_hours_ago = timestamp + timedelta(hours=-2)
        foods = [food[1:] for food in food_data if
                 timestamp >= food[0] > two_hours_ago]

    # normalize all nutrients to grams
    food_intake = []
    for food in foods:
        current_food = replace_none(food[:2]) if len(food) > 1 else []  # add glycemic index and calories
        for j in range(2, len(food), 2):  # start at index 2 so we skip the columns that don't have measurement labels
            nutrient = conversions[food[j + 1]](food[j])  # j+1 is the measurement for the current nutrient
            current_food.append(nutrient)

        food_intake.append(current_food)

    # sum all foods consumed in a "meal" by its nutrients
    food_array = [sum(item) for item in zip(*food_intake)]

    if not food_array:
        food_array = [0] * 26

    return food_array


def get_activity_change(bg_data: tuple, activity_data: list, i: int, timestamp: datetime) -> list:
    """
    Gets the change in activity levels between ticks of blood glucose.

    Note - It is possible we may want to look at the cumulative activity within a given window.
    """
    if i == 0:
        # get a base line of activity if this is the first bg level
        # remove the receive date and date columns with activity[2:]
        activities = [activity[2:] for activity in activity_data if activity[0] <= timestamp]
    else:
        # get change in activity between this bg level and the last
        # remove the receive date and date columns with activity[2:]
        last_timestamp, _ = bg_data[i - 1]
        activities = [activity[2:] for activity in activity_data if
                      timestamp >= activity[0] > last_timestamp]

    # sum over each record to get cumulative change in activity
    activity_change = [sum(replace_none(ac)) for ac in zip(*activities)]

    if not activity_change:
        activity_change = [0] * 6

    return activity_change


def replace_none(input_list) -> list:
    """
    Replaces instances of None with zero in given list
    """
    return [i if i is not None else 0 for i in input_list]


def convert_micrograms_to_grams(value):
    return value / 1000000


def convert_milligrams_to_grams(value):
    return value / 1000


def convert_nochange(value):
    return value


conversions = {
    'µg': convert_micrograms_to_grams,
    'mcg': convert_micrograms_to_grams,
    'mg': convert_milligrams_to_grams,
    'g': convert_nochange,
    None: convert_nochange
}

correlated_data = []
data_dict = get_data()
correlated_data = correlated_data + correlate_data(data_dict)

cols = ['timestamp',
        'label',
        'bg',
        'bg-5',
        'bg-10',
        'bg-15',
        'bg-20',
        'bg-25',
        'bg-30',
        'steps',
        'sedentary_minutes',
        'lightly_active_minutes',
        'fairly_active_minutes',
        'very_active_minutes',
        'calories_out',
        'glycemicindex',
        'calories',
        'calcium',
        'carbs',
        'fiber',
        'folate',
        'iron',
        'magnesium',
        'monounsaturatedfat',
        'niacin',
        'phosphorus',
        'polyunsaturatedfat',
        'potassium',
        'protein',
        'riboflavin',
        'saturatedfat',
        'sodium',
        'sugar',
        'thiamin',
        'totalfat',
        'vitamina',
        'vitaminb6',
        'vitaminc',
        'vitamine',
        'vitamink',
        'zinc',
        'total_minutes_of_sleep',
        'basal_insulin',
        'bolus_insulin']
test_data = pd.DataFrame(correlated_data, columns=cols)

#df.to_csv(path_or_buf='c:/users/sunil/downloads/bgdata.csv', index=False)
