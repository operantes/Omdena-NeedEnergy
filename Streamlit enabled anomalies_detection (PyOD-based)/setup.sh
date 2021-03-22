
set -x
DIR_STREAMLIT='~/.streamlit'
mkdir -p $DIR_STREAMLIT

ls -la $DIR_STREAMLIT

sed 's/\$PORT/'$PORT'/' > $DIR_STREAMLIT/config.toml <<EOF2
[server]
headless = true
enableCORS=false
port = $PORT
EOF2

cat > $DIR_STREAMLIT/credentials.toml <<EOF
[general]
email = 'alvaro.montesino@gmail.com' 
EOF

ls -la $DIR_STREAMLIT
nl -ba $DIR_STREAMLIT/*.toml
