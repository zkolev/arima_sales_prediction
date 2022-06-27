from app import app

if __name__ == '__main__':
        app.run_server(port=8052, debug=True, use_reloader=False)
        # server.run(port=8052, debug=True) # For gunicorn
