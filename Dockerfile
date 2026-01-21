# Stage 1: Builder - Install dependencies
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV NUMBA_DISABLE_CACHING=1

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Dev - For running tests
FROM builder as dev

ENV APP_HOME=/home/app/web
WORKDIR $APP_HOME

# Copy the application and test code
COPY ./learning_agent ./learning_agent
COPY ./tests ./tests

# Set the default command to run tests
CMD ["pytest"]


# Stage 3: Final - Create the production image
FROM python:3.12-slim

# Create a non-root user and group
RUN addgroup --system app && adduser --system --group app

# Set the home directory for the new user
ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY ./learning_agent ./learning_agent

# Set the path to use the virtual environment's executables
ENV PATH="/opt/venv/bin:$PATH"

# Change ownership of the app directory
RUN chown -R app:app $APP_HOME

# Switch to the non-root user
USER app

# Expose the port the app runs on
EXPOSE 8004

# Run the application
CMD ["uvicorn", "learning_agent.main:app", "--host", "0.0.0.0", "--port", "8004"]
