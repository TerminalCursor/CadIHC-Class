def get_datestring():
    from datetime import datetime

    now = datetime.now()
    return f"{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-{now.microsecond:06}"


class Logger:
    db_name = None
    db_category = None
    db_lr = None
    db_loss = None

    def __init__(self, base_path, echo=True):
        from os.path import exists

        assert exists(base_path)
        self.enabled = True
        self.echo = echo
        self.timestamp = get_datestring()
        self.log_db = f"{base_path}/Results.db"
        self.db_data = {}
        self.filename = f"{base_path}/{self.timestamp}.txt"
        # with open(self.filename, "w+") as fhandle:
        #    fhandle.write("")

    def meta(self, name, category, lr, loss):
        self.db_name = name
        self.db_category = category
        self.db_lr = lr
        self.db_loss = loss

    def add(self, formatted_text):
        if isinstance(formatted_text, dict):
            import json

            return self.add(json.dumps(formatted_text, indent=4))
        if isinstance(formatted_text, tuple):
            import json

            return self.add(json.dumps(formatted_text, indent=4))
        if self.enabled:
            with open(self.filename, "a") as fhandle:
                if self.echo:
                    print(formatted_text, end="")
                fhandle.write(formatted_text)
        return self

    def addnl(self, formatted_text):
        self.add(formatted_text)
        if self.enabled:
            with open(self.filename, "a") as fhandle:
                fhandle.write("\n")
            if self.echo:
                print()
        return self

    def assoc(self, **res_dict):
        for key in res_dict.keys():
            if key.lower() in [
                "total_accuracy",
                "training_accuracy",
                "validation_accuracy",
                "total_accuracy_c",
                "training_accuracy_c",
                "validation_accuracy_c",
            ]:
                self.db_data[key.lower()] = float(res_dict[key])
            elif key.lower() in ["total_raw", "training_raw", "validation_raw"]:
                self.db_data[key.lower()] = res_dict[key]

    def dump(self):
        import sqlite3

        con = sqlite3.connect(self.log_db)
        cur = con.cursor()
        cur.execute(f"DELETE FROM RESULTS WHERE TIMESTAMP = '{self.timestamp}'")
        query1 = "INSERT INTO RESULTS( "
        query2 = ") VALUES ("
        query3 = ");"
        # Timestamp
        query1 += f"TIMESTAMP, "
        query2 += f"'{self.timestamp}', "
        if self.db_name is not None:
            assert type(self.db_name) is str
            query1 += f"NAME, "
            query2 += f"'{self.db_name}', "
        if self.db_category is not None:
            assert type(self.db_category) is str
            query1 += f"CATEGORY, "
            query2 += f"'{self.db_category}', "
        if self.db_lr is not None:
            assert type(self.db_lr) is float
            query1 += f"LR, "
            query2 += f"{self.db_lr}, "
        if self.db_loss is not None:
            assert type(self.db_loss) is str
            query1 += f"LOSS, "
            query2 += f"'{self.db_loss}', "
        for key in ["total_raw", "training_raw", "validation_raw"]:
            if key in self.db_data.keys():
                query1 += f"{key.upper()}, "
                data = (
                    str(self.db_data[key])
                    .replace("\n", "")
                    .replace("tensor", "")
                    .replace(" ", "")
                )
                query2 += f"'{data}', "
        for key in [
            "total_accuracy",
            "training_accuracy",
            "validation_accuracy",
            "total_accuracy_c",
            "training_accuracy_c",
            "validation_accuracy_c",
        ]:
            if key in self.db_data.keys():
                query1 += f"{key.upper()}, "
                query2 += f"{self.db_data[key]}, "
        query1 = query1[:-2] + " "
        query2 = query2[:-2] + " "
        if self.echo:
            print(query1 + query2 + query3)
        con.execute(query1 + query2 + query3)
        con.commit()
        con.close()
