import streamlit as st
import discord
import model

from discord.ext import commands

app = commands.Bot(command_prefix='/')
 
@app.event
async def on_ready():
    print('Done')
    await app.change_presence(status=discord.Status.online, activity=None)

@app.command()
async def hello(ctx):
    await ctx.send('Hello I am Bot!')
    
app.run('MTEzNDM5NDY0MTUwNTA2NzAxOQ.G3_Ucr.9MCbN3gUiAj8UT9MrowMahpJmb8CiapDFnidUU')

model.printTFVersion()

st.set_page_config(
    page_icon="🎹",
    page_title="LOFIAI",
    layout="wide",
)

st.header("LOFI-LOOP 🎷")
st.subheader("made by mmmosd")