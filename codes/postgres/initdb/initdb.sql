/*******************************************************************************
   Create Tables
********************************************************************************/
CREATE TABLE "log_logins"
(
    "id" SERIAL PRIMARY KEY,
    "username" VARCHAR(120) NOT NULL,
    "is_successful" BOOLEAN NOT NULL,
    "created_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "log_llm_inputs_outputs"
(
    "id" SERIAL PRIMARY KEY,
    "input" VARCHAR(120) NOT NULL,
    "output" VARCHAR(120) NOT NULL,
    "created_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "user_feedbacks"
(
    "id" SERIAL PRIMARY KEY,
    "scale" INTEGER NOT NULL,
    "feedback" TEXT NOT NULL,
    "created_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);