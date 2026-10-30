FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
COPY requirements.txt ./dashboard/requirements.txt
RUN python -m pip install --no-cache-dir -r dashboard/requirements.txt
COPY . ./dashboard
COPY .streamlit /app/.streamlit
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
