
import axios from "axios"

async function postData(url = "", data = {}) {
    // Default options are marked with *
    const response = await fetch(url, {
      method: "POST", // *GET, POST, PUT, DELETE, etc.
      mode: "no-cors", // no-cors, *cors, same-origin
      cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
      credentials: "same-origin", // include, *same-origin, omit
        headers: {
            "Content-Type": "application/json",
            // 'Content-Type': 'application/x-www-form-urlencoded',
        },
      redirect: "follow", // manual, *follow, error
      referrerPolicy: "no-referrer", // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify(data), // body data type must match "Content-Type" header
    });
    return await response.json(); // parses JSON response into native JavaScript objects
}

async function getData(url = "") {
    // Default options are marked with *
    const response = await fetch(url, {
      method: "GET", // *GET, POST, PUT, DELETE, etc.
      mode: "no-cors", // no-cors, *cors, same-origin
    });
    return await response.json(); // parses JSON response into native JavaScript objects
}

const oF = axios.create({
    baseURL: 'http://localhost:8000/',
    timeout: 300000,
    withCredentials: false,
    // headers: {'X-Custom-Header': 'foobar'}
});


export default class API {
    static async fnGetFilterOptions() {
        return (await oF.get(`get_filter_options`)).data
    }

    static async fnProcessChatMessage(oArgs) {
        return (await oF.post(`process_chat_message`, oArgs)).data
    }
}