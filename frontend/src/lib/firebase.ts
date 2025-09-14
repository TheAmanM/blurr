import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyB9cXojIKf2SsXjb69UedLHPipy7x6f-S8",
  authDomain: "web-rtc-demo-95be5.firebaseapp.com",
  projectId: "web-rtc-demo-95be5",
  storageBucket: "web-rtc-demo-95be5.firebasestorage.app",
  messagingSenderId: "61059022514",
  appId: "1:61059022514:web:6b338e9a51166ec0d793ab",
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
