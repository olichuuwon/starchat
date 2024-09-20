/*******************************************************************************
   Create Tables
********************************************************************************/
CREATE TABLE "log_logins"
(
    "id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    "is_successful" BOOLEAN NOT NULL,
    "username" TEXT NULL,
    "token" TEXT NULL,
    "error_message" TEXT NULL,
    "created_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "log_llm_inputs_outputs"
(
    "id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    "username" TEXT NULL,
    "input" TEXT NULL,
    "output" TEXT NULL,
    "created_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "user_feedbacks"
(
    "id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    "username" TEXT NULL,
    "scale" INTEGER NOT NULL,
    "feedback" TEXT NOT NULL,
    "created_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);