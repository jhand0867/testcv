import streamlit as st
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time

from jax.experimental.pallas.ops.tpu.example_kernel import double

#from streamlit import columns, title

st.header("Hello World")
st.write("This is a test for Python Streamlit ")
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [1, 2, 3])
# other plotting actions...
st.pyplot(fig)
st.write("First attempt to display table data")

# streamlit write with pandas dataframe - interactive frame
st.write(pd.DataFrame({'first column':[1,2,3,4],
                       'second column':[10,20,30,40]}))
# streamlit table same dataframe - read only
st.table(pd.DataFrame({'first column':[1,2,3,4],
                       'second column':[10,20,30,40]}))

# streamlit write with numpy
# create a dataframe with random data
df = pd.DataFrame(np.random.randn(10,20),dtype=np.float64,
                  columns=('Col %d' % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=0))
st.table(df)
st.line_chart(df)

chart_data = pd.DataFrame(np.random.randn(20,3),columns=['a','b','c'])
st.line_chart(chart_data)

# plot a map
map_data = pd.DataFrame(np.random.randn(1000,2)/[50,50] + [37.76,-122.4],
                        columns=['lat','lon'])
st.map(map_data)

# slider
x = st.slider('x',min_value=1, max_value=30000)  # 👈 this is a widget
st.write(x, 'squared is', x * x)

# text input
i = st.text_input("Enter number of series", help="series 1 to 4", key="numofseries" )

# checkbox to show/hide data
st.sidebar.title("Select series to include")
seriesA =st.sidebar.checkbox("Include series A", value=1)
seriesB =st.sidebar.checkbox("Include series B", value=1)
seriesC =st.sidebar.checkbox("Include series C", value=1)
seriesD =st.sidebar.checkbox("Include series D", value=1)

include = seriesA + seriesB + seriesC

#
if st.checkbox('Show dataframe'):
    if include == 1:
        columnar = ['a']
    elif include == 2:
        columnar = ['a', 'b']
    else:
        columnar = ['a', 'b', 'c']

    chart_data = pd.DataFrame(
        np.random.randn(20, include),columns=columnar)

    st.bar_chart(chart_data)

add_select_box = st.sidebar.selectbox("How would you like us to contact you?",('no','email','phone'))

# show progress of a long process
latest_iteration = st.empty()

'Starting a long computation...'
pbar = st.progress(0)

for i in range(100):
    # update progress every interaction
    latest_iteration.text(f'Iteration: {i+1}')
    pbar.progress(i+1)
    time.sleep(0.1)

'We are done !'

# using session.state
'''
    anything in the state will persist reruns (NO refresh)
    Use st.session.state.<varible_name> without the brackets 
'''
if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")

# simulation caching with st.session_state
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)

# connections


  # - Can use st.cache_resource to cache connections
  #   - most common
  #   - allow to connection to connect with any python library
  # - Can use st.connection to connect to the most common databases
  # - configuration is at
  #   - C:\Users\jhand\.streamlit or
  #   - <project dir>\.streamlit\
  #   - file needs to be created if not present secretes.toml


# conn = st.connection("my_database")
# df = conn.query("select * from table1")
# st.dataframe(df)

