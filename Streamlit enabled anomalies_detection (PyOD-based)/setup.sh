
set -x
DIR_STREAMLIT='~/.streamlit'
mkdir -p $DIR_STREAMLIT

ls -la ~/.
env

cat > $DIR_STREAMLIT/config.toml <<EOF
[server]
headless = true
enableCORS = false
port = $PORT

EOF

cat > $DIR_STREAMLIT/credentials.toml <<EOF
[general]
email = alvaro.montesino@gmail.com 

EOF

chmod -R ug+r $DIR_STREAMLIT
ls -la $DIR_STREAMLIT
nl -ba $DIR_STREAMLIT/*.toml
