# ---------------------------------------------------------------------------- #
#                                    DEFINS                                    #
# ---------------------------------------------------------------------------- #
NAME	:= inquisitor.py
TARGET	:= inquisitor

SERVER	:= server
CLIENT	:= client

CYAN="\033[1;36m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[m"

# ---------------------------------------------------------------------------- #
#                                     RULES                                    #
# ---------------------------------------------------------------------------- #

.PHONY: all
all	:
	docker compose up --build -d && docker exec -it python /bin/bash

.PHONY: clean
clean	:
	docker compose down

.PHONY: prune
prune	: clean
	docker system prune -f -a

.PHONY: fclean
fclean	: clean
	-docker stop $(shell docker ps -qa) 2>/dev/null
	-docker rm $(shell docker ps -qa) 2>/dev/null
	-docker rmi -f $(shell docker images -qa) 2>/dev/null
	-docker volume rm $(shell docker volume ls -q) 2>/dev/null
	-docker network rm $(shell docker network ls -q) 2>/dev/null

.PHONY: build
build	:
	docker compose up --build -d

.PHONY: exec
exec	: build
	docker exec -it python /bin/bash

.PHONY: re

# ---------------------------------------------------------------------------- #
#                                     UTILS                                    #
# ---------------------------------------------------------------------------- #

.PHONY: run
