import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#generating demand
np.random.seed(42)
top, left = 13.049651231078998, 77.4915600797753
bottom, right = 12.867433941265721, 77.74402207737981
num_points = 4500
random_latitudes = np.random.uniform(low=bottom, high=top, size=num_points)
random_longitudes = np.random.uniform(low=left, high=right, size=num_points)
demand_df = pd.DataFrame({"lat" : random_latitudes, "long" : random_longitudes})

#creating clusters
kmeans = KMeans(n_clusters=15, random_state=42)
demand_df["cluster"] = kmeans.fit_predict(demand_df)
demand_df["area_name"] = "area" + demand_df.cluster.astype(str)
demand_df["customer"] = demand_df[["lat", "long"]].index

#sampling points for Charging station
stations = demand_df.groupby('area_name').apply(lambda x: x.sample(4, random_state=42)).reset_index(drop=True)
stations["visibility"] = stations.groupby('cluster').cumcount(ascending=False) + 1
stations["cs_name"] = stations.area_name + "_c_" + stations.visibility.astype(str)
stations.rename(columns={"lat" : "cs_lat", "long" : "cs_long"}, inplace=True)
stations = stations[["cs_name", "cs_lat", "cs_long", "area_name", "visibility"]]

#modeling as a MILP
model_input = demand_df.groupby('cluster').lat.count().reset_index(name="demand") 
model_input = pd.concat([model_input]*4, ignore_index=True)
model_input["visibility"] = model_input.groupby('cluster').cumcount(ascending=False) + 1
model_input['fcost'] = model_input['visibility'] + model_input['demand']/50 + np.random.uniform(3, 6, len(model_input))
model_input["daily_demand"] = model_input["demand"]//7
model_input["area_name"] = "area" + model_input.cluster.astype(str)

#IP model
from pulp import *
area_list = model_input.area_name.unique()
visib_list = model_input.visibility.unique()
pairs = [(area, visib) for area in area_list for visib in visib_list] 
fixed_cost = model_input.groupby('area_name').apply(lambda x: x.set_index('visibility')['fcost'].to_dict()).to_dict() #converting to dictionary for significantly faster compute
area_wise_demand = model_input.drop_duplicates(subset=['area_name']).set_index("area_name")["daily_demand"].to_dict()
model = LpProblem("charging_stations_location", LpMinimize)
mapping_var = LpVariable.dicts("station_",(area_list,visib_list),0,3,LpInteger) 
mapping_var_binary = LpVariable.dicts("station_f_",(area_list,visib_list),0,1,LpBinary) 
model += lpSum(fixed_cost[area][vis] * mapping_var_binary[area][vis] for (area,vis) in pairs) 
model += lpSum(2 * mapping_var[area][vis] for (area,vis) in pairs)
for area in area_list:
    model+= lpSum(15 * mapping_var[area][vis] for vis in visib_list) >= area_wise_demand[area] 
    for vis in visib_list:
        model+= mapping_var_binary[area][vis] >= mapping_var[area][vis] * 0.001
model+= lpSum(mapping_var[area][4] for area in area_list) >=  4
model+= lpSum(mapping_var[area][3] for area in area_list) >=  2
solver = PULP_CBC_CMD(mip=True, msg=0, timeLimit=1200)
model.solve(solver)

#final solution
df_sol = pd.DataFrame.from_dict(mapping_var, orient="index")
df_sol = df_sol.map(lambda x: value(x))
df_sol["area_name"] = df_sol.index
df_sol = df_sol.melt(id_vars="area_name")
df_sol.rename(columns={"variable" : "visibility", "value":"n_stations"}, inplace=True)
df_sol = pd.merge(model_input, df_sol, on=["area_name", "visibility"])
stations = pd.merge(stations, df_sol[["visibility", "area_name", "n_stations"]], on=["visibility", "area_name"])
stations["open"] = stations.n_stations.apply(lambda x: "Open" if x>0 else "Closed")

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np  
from scipy.spatial import ConvexHull

def plotter_fxn(stations, demand_df):
    mapbox_api_token = "INSERT_TOKEN_HERE"
    fig = go.Figure()
    for group, group_df in demand_df.groupby('area_name'):
        if len(group_df)<=2:
            continue
        hull = ConvexHull(group_df[['long', 'lat']])
        hull_x = group_df['long'].values[hull.vertices]
        hull_y = group_df['lat'].values[hull.vertices]
        # group_color = 'rgba(205, 67, 67, 0.3)'
        group_color = 'rgba(87, 211, 155, 0.3)'
        fig.add_trace(go.Scattermapbox(
            lon=hull_x, 
            lat=hull_y, 
            mode='lines', 
            fill='toself', 
           marker={
               'color' : "black",
               'size' : 10
           },
           fillcolor=group_color
        ))
        fig.add_trace(
        go.Scattermapbox(
            lon=stations['cs_long'],
            lat=stations['cs_lat'],
            mode='markers',
            marker=dict(
                size=5*stations['n_stations'].apply(lambda x: x+1),
                color=stations['visibility'].map({1: '#028cba', 2: '#02ba64', 3: '#c9ac08' ,4:'#0fd612'})

            ),
        )
        )
    fig.update_layout(
        mapbox = {
            'style': 'open-street-map',
            'center' : {
                'lon' : np.mean(demand_df['long']),
                'lat' : np.mean(demand_df['lat']) 
            },
            'zoom' : 11
        },
        showlegend = False
        )
    fig.update_layout(mapbox_style="dark", mapbox_accesstoken=mapbox_api_token)
    return fig
fig = plotter_fxn(stations, demand_df)

fig.show()
