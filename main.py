from app import app

if __name__ == '__main__':
        server = app.server
        # server.run(port=8052, debug=True)
        server.run(port=8052, debug=True) # For gunicorn
