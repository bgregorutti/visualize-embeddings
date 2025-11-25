FROM nginx:alpine

COPY --chmod=644 index.html /usr/share/nginx/html/
COPY --chmod=644 app.js /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
