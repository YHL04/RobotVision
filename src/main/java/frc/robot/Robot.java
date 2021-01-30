package frc.robot;

import edu.wpi.first.wpilibj.TimedRobot;

import java.lang.Runtime;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;


public class Robot extends TimedRobot {
  BufferedReader stdInput;
  BufferedReader stdError;

  @Override
  public void robotInit() {
    new Thread(
      () -> {
        String s = null;
        try {
          Process p = Runtime.getRuntime().exec(
          "C:\\Users\\22lim\\AppData\\Local\\Programs\\Python\\Python37\\python.exe test.py");

          stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
          stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));

          while ((s = stdInput.readLine()) != null) {
            System.out.println(s);
          }

          while ((s = stdError.readLine()) != null) {
            System.out.println(s);
          }

        } catch (IOException e) {
          e.printStackTrace();
          System.exit(-1);
        }
      }).start();
  }
}